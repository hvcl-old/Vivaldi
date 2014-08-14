from mpi4py import MPI

# mpi init
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

def disconnect():
	print "AA", rank, name
	comm.Barrier()
	comm.Disconnect()
	exit()

### GPU_unit description
### GPU unit manage everything related GPU charged
### every data is numpy and it saved in GPU memory

from Vivaldi_misc import *
from Vivaldi_memory_packages import Data_package
from Vivaldi_load import *


data_list = {}
block_memory = {}
block_maximum = {}
block_written = {}

mmtx = None


def send_data_package(data_package, dest, tag):
	dp = data_package
	t_data, t_devptr = dp.data, dp.devptr
	dp.data, dp.devptr = None, None
	comm.isend(dp, dest=dest, tag=tag)
	dp.data, dp.devptr = t_data, t_devptr
	t_data,t_devptr = None, None

def idle():
	rank = comm.Get_rank()
	comm.isend(rank,		dest=2,		tag=2)
	comm.isend("idle",	dest=2,		tag=2)

def notice(data_package):
	rank = comm.Get_rank()
	comm.isend(rank,			dest=2,	tag=2)
	comm.isend("notice",		dest=2,	tag=2)

	data = data_package.data
	devptr = data_package.devptr
	
	send_data_package( data_package, dest=2, tag=21)

global requests
requests = []

def send(data, data_package, dest=None, gpu_direct=False):
	tag = 52
	# send data_package
	dp = data_package
	send_data_package(dp, dest, tag)

	if log_type in ['time','all']: st = time.time()

	request = comm.Isend(data, dest=dest, tag=57)

	if VIVALDI_BLOCKING: MPI.Request.Wait(request)
	else: requests.append((request, data))

	if log_type in ['time','all']:
		u = dp.unique_id
		bytes = dp.data_bytes
		t = MPI.Wtime()-st
		ms = 1000*t
		bw = bytes/GIGA/t
		name = dp.data_name

		log("rank%d, \"%s\", u=%d, from rank%d to rank%d using MPI memory transfer, Bytes: %.3fMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, name, u, rank, dest, bytes/MEGA, ms, bw),'time', log_type)

def recv():
	data_package = comm.recv(source=source,	tag=52)
	dp = data_package
	data_dtype = dp.data_dtype

	data_memory_shape = dp.data_memory_shape
	dtype = dp.data_contents_memory_dtype
	data = numpy.empty(data_memory_shape, dtype=dtype)
	request = comm.Irecv(data, source=source, tag=57)
	requests.append(request)
	# Vivaldi reader cannot be nonblocking receive
	# because next operation is save to hard disk, and we need valid data for save
	MPI.Request.Wait(request)

	return data, data_package

def save_data(data, data_package):
	# if data is numpy.ndarray, copy to GPU and save only devptr
	dp = data_package
	dp.data = data
	dp.memory_type = 'memory'

	notice(dp)
	
	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
	if u not in data_list: data_list[u] = {}
	if str(ss) not in data_list[u]:data_list[u][str(ss)] = {}
	data_list[u][ss][sp] = dp

def get_file_pointer(file_name):
	f = open(file_name,'r')

	while True:
		line = f.read_line()
		if f == '':break

	return 

def copyto(dest, source):
	i = 0
	for elem in source:
		dest[i] = elem
		i += 1

def load_data(data_package, data_range):

	if log_type in ['time','all']: st = time.time()
	dp = data_package
	file_name = dp.file_name

	p = file_name.find('.')
	extension = file_name[p+1:].lower()

	f = open(file_name,'rb')

	# creat data for memroy read
	chan = dp.data_contents_memory_shape
	shape = range_to_shape(data_range)
	data_memory_shape = list(shape)
	if chan != [1]: data_memory_shape = list(shape) + chan
	data_contents_memory_dtype = dp.data_contents_memory_dtype

	length = 1
	for elem in data_memory_shape:
		length *= elem

	file_size = {}
	i = 0
	for axis in AXIS[::-1]:
		if axis in data_range:
			file_size[axis] = dp.full_data_range[axis][1]-dp.full_data_range[axis][0]
		else:
			file_size[axis] = 0
		i += 1


	# index of start point
	data_start = {}
	for axis in AXIS:
		if axis not in data_range:	data_start[axis] = 0
		else: data_start[axis] = data_range[axis][0]

	dim = len(data_range)
	if dim == 2:
		W = file_size['w']
		Z = file_size['z']
		Y = file_size['y']
		X = file_size['x']
		
		ss = ''
		idx = data_start['y']*X
		file_python_dtype = Vivaldi_dtype_to_python_dtype(dp.file_dtype)
		file_bytes = get_bytes(file_python_dtype)

		for y in range(data_range['y'][0], data_range['y'][1]):
			idx = y * (X) * file_bytes
			f.seek(idx,0)
			ts = f.read((X) * file_bytes)
			ss += ts

		buf = numpy.fromstring(ss, dtype=file_python_dtype)
		height = data_range['y'][1]-data_range['y'][0]
		buf = buf.reshape((height,X))
		buf = buf[:, data_range['x'][0]:data_range['x'][1]]

		buf = buf.astype(data_contents_memory_dtype)
	
	elif dim == 3:
		W = file_size['w']
		Z = file_size['z']
		Y = file_size['y']
		X = file_size['x']
		
		ss = ''
		idx = data_start['z']*X*Y
		file_python_dtype = Vivaldi_dtype_to_python_dtype(dp.file_dtype)
		file_bytes = get_bytes(file_python_dtype)*chan[0]

		for z in range(data_range['z'][0], data_range['z'][1]):
			idx = z * (X*Y) * file_bytes
			f.seek(idx,0)
			ts = f.read((X*Y) * file_bytes)
			ss += ts

		buf = numpy.fromstring(ss, dtype=file_python_dtype)
		depth = data_range['z'][1]-data_range['z'][0]

		if chan == [1]:
			buf = buf.reshape((depth,Y,X))
		else:
			buf = buf.reshape((depth,Y,X,chan[0]))

		buf = buf[:,data_range['y'][0]:data_range['y'][1], data_range['x'][0]:data_range['x'][1]]
		buf = buf.astype(data_contents_memory_dtype)

	if log_type in ['time','all']:
		u = dp.unique_id
		bytes = buf.nbytes
		t = MPI.Wtime()-st
		ms = 1000*t
		bw = bytes/GIGA/t
		name = dp.data_name

		log("rank%d, \"%s\", u=%d, from harddisk to Reader CPU memory, Bytes: %.3fMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, name, u, bytes/MEGA, ms, bw),'time', log_type)


	return buf

def data_finder(u, ss, sp, gpu_direct=False, data_range=None):
	source_package = data_list[u][ss][sp]
	if data_range == None:
		data_range = source_package.full_data_range

	cut = False
	if data_range != None and data_range != source_package.data_range: cut = True

	out_of_core = source_package.out_of_core
	if out_of_core:
		data = load_data(source_package, data_range)
		data_halo = 0
	else:
		if cut:
			data = source_package.data
			source_range = source_package.data_range
			cmd = '['
	
			for axis in AXIS[::-1]:
				if axis in data_range:
					cmd += '%d:%d,'%(data_range[axis][0], data_range[axis][1])
			cmd = 'data = numpy.array(data'+cmd[:-1]+'])'

			exec cmd
			data_halo = 0
		else:
			data = source_package.data
			data_halo = source_package.data_halo

	source_package.memory_type = 'memory'	
	return data, source_package, data_halo

def memcpy_p2p_send(task, dest_rank, data, halo_size):
	wr = task.work_range

	# say dest to prepare to recv this data
	comm.isend(rank,					dest=dest_rank,	tag=5)
	comm.isend("memcpy_p2p_recv",		dest=dest_rank,	tag=5)
	comm.isend(task,					dest=dest_rank,	tag=57)
	comm.isend(halo_size,				dest=dest_rank,	tag=57)

	dest_gpu_direct = False
	# send data
	source_package = task.source

	data_package = source_package.copy()
	dp = data_package
	
	dp.data_dtype = numpy.ndarray
	dp.memory_type = 'memory'
	dp.set_data_range(wr)
	dp.set_buffer_range(wr)
	dp.buffer_halo = 0
	
	send(data, dp, dest=dest, gpu_direct=False)

	buf = None
	dest_devptr = None	

def save_image(buf=None, chan=None, file_name=None, extension=None, normalize=False):
	st = time.time()
	img = None

	# if normalize is true, than normalize from 0 ~ 255, because image is 8 bit
	if normalize == True:
		min = buf.min()
		max = buf.max()
		if min != max:
			buf = (buf-min)*255/(max-min)
	
	buf = buf.astype(numpy.uint8)
	min = buf.min()
	max = buf.max()
	if chan == [1]:   img = Image.fromarray(buf, 'L')
	elif chan == [3]: img = Image.fromarray(buf, 'RGB')
	elif chan == [4]: img = Image.fromarray(buf, 'RGBA')
	e = os.system("mkdir -p result")
	img.save('./result/%s'%(file_name+'.'+extension), format=extension)

	time.sleep(2)
	if log_type in ['time','all']:
		t = time.time()-st
		ms = 1000*t
		sp = buf.nbytes/MEGA
		bytes = buf.nbytes
		log("rank%d, \"%s\", save time to hard disk bytes: %.3fMB %.3f ms %.3f MBytes/sec"%(rank, file_name, bytes/MEGA, ms, sp),'time',log_type)

global cnt
cnt = 0

def memcpy_p2p_recv(task, halo):
	data, data_package = recv()
	dp = task.dest

#	print "SOURCE", dp, source
	#assert(False)
	chan = dp.data_contents_memory_shape

	file_name = dp.file_name
	extension = dp.extension
	halo = dp.data_halo

	if extension in ['png','jpg','jpeg'] or file_name == '':
		if file_name == '':
			file_name += dp.data_name + str(dp.unique_id)
			extension = 'png'
		if dp.out_of_core:
			file_name += dp.split_shape + dp.split_position

		normalize = dp.normalize
		save_image(buf=data, chan=chan, file_name=file_name, extension=extension, normalize=normalize)
	else:

		# if normalize is true, normalize from 0 to 1
		if dp.normalize:
			min = data.min()
			max = data.max()
			if min != max:
				data = (data-min)/(max-min)*1

		# we have data
		# extension
		if extension == 'raw':
			global cnt
			e = os.system("mkdir -p result/%s"%(file_name))
			f = open('./result/%s/%s%d.raw'%(file_name,file_name, cnt), 'wb')
			f.write(data)
			f.close()
			cnt += 1

		return

# Computing_unit related function
#####################################################################################################

# memory manager related function
#####################################################################################################
def mem_release(data_package):
	comm.isend(rank,         dest=2,     tag=2)
	comm.isend("release",    dest=2,     tag=2)
	
	send_data_package(data_package, dest=2, tag=27)
		
#def mem_retain(data_package):
#	comm.isend(rank,         dest=2,     tag=2)
#	comm.isend("retain",     dest=2,     tag=2)
#	comm.isend(data_package, dest=2,     tag=28)

# main 
#####################################################################################################

# initialize variables

# log type
# default is nothing
if '-L' in sys.argv:
	idx = sys.argv.index('-L')
	log_type = sys.argv[idx+1]

# gpudirect 
# default is true, but we have to check machine state
GPUDIRECT = True
if '-G' in sys.argv:
	idx = sys.argv.index('-G')
	GPUDIRECT = sys.argv[idx+1]
	if GPUDIRECT == 'on':GPUDIRECT = True
	else: GPUDIRECT = False


# data transaction is blocking or nonblocking
if '-B' in sys.argv:
	VIVALDI_BLOCKING = False
	if '-B' in sys.argv:
	    idx = sys.argv.index('-B')
	    VIVALDI_BLOCKING = sys.argv[idx+1]
	    if VIVALDI_BLOCKING.lower() in ['true','on']: VIVALDI_BLOCKING = True
	    elif VIVALDI_BLOCKING.lower() in ['false','off']: VIVALDI_BLOCKING = False


log("rank%d, Reader launch at %s "%(rank, name),'general',log_type)

#comm.isend("device_available", dest=0, tag=777)

flag_times = {}
for elem in ["recv","send_order","free","memcpy_p2p_send","wait","cutting"]:
	flag_times[elem] = 0

flag = ''
while flag != "finish":
	st = time.time()
	log("rank%d, Reader Waiting "%(rank),'general',log_type)
	source = comm.recv(source=MPI.ANY_SOURCE, tag=5)
	flag = comm.recv(source=source, tag=5)
	flag_times['wait'] += time.time() - st
	log("rank%d, Reader %s"%(rank, flag),'general',log_type)


	if flag == "synchronize":
		# synchronize
		comm.send("Done", dest=source, tag=999)

	elif flag == "recv":
		st = time.time()
		data, data_package = recv()
		dp = data_package
		save_data(data, dp)
#		mem_retain(dp)
		idle()
		flag_times[flag] += time.time() - st
	elif flag == "send_order":
		st = time.time()
		dest = comm.recv(source=source, tag=53)
		u = comm.recv(source=source, tag=53)
		ss = comm.recv(source=source, tag=53)
		sp = comm.recv(source=source, tag=53)
		log("rank%d, order to send data u=%d to %d"%(rank, u, dest),'general',log_type)

		gpu_direct = False
		data, data_package,_ = data_finder(u,ss,sp, gpu_direct=gpu_direct)

		send(data, data_package, dest=dest)
		comm.isend(True, dest=2 ,tag=53)

		idle()
		flag_times[flag] += time.time() - st

	elif flag == "memcpy_p2p_send":
		st = time.time()
		dest = comm.recv(source=source, tag=56)
		task = comm.recv(source=source, tag=56)
		u = task.source.unique_id
		ss = task.source.split_shape
		sp = task.source.split_position

		data, _, halo_size = data_finder(u,ss,sp, data_range=task.work_range)
				
		memcpy_p2p_send(task, dest, data, halo_size)
		idle()
		flag_times[flag] += time.time() - st

	elif flag == "memcpy_p2p_recv":
		task = comm.recv(source=source, tag=57)
		halo_size = comm.recv(source=source, tag=57)
		memcpy_p2p_recv(task, halo_size)
#		send_data_package(task.source)
		mem_release(task.source)
		idle()

	elif flag == "free":
		st = time.time()
		u = comm.recv(source=source, tag=58)

		del(data_list[u])
		flag_times[flag] += time.time() - st

	elif flag == "notice_data_out_of_core":
		data_package = comm.recv(source=source, tag=501)
		dp = data_package
		save_data(None, dp)
	
	elif flag == "prepare_out_of_core_save":
		data_package = comm.recv(source=source, tag=502)
		dp = data_package

		file_name = dp.file_name
		idx = file_name.find('.')
		extension = file_name[idx+1:]

		el = extension.lower()
	
		
		if el in ['png','jpg','jpeg']:
			pass
		elif el == 'sd':
			file_name = file_name[:idx]
			e = os.system("mkdir -p result")

			f = open('./result/%s.sd'%(file_name),'w')
			f.write("filename: %s\n"%(file_name))
			sfw = ''
			for elem in dp.full_data_shape:
				sfw += str(elem) + ' '

			f.write('shape: %s\n'%(sfw))
			f.write('chan: %s\n'%(str(dp.data_contents_memory_shape[0])))
			Vivaldi_memory_dtype = python_dtype_to_Vivaldi_dtype(dp.data_contents_memory_dtype)
			f.write('dtype: %s\n'%(Vivaldi_memory_dtype))
			f.close()

		elif el == 'dat':
			file_name = file_name[:idx]
			e = os.system("mkdir -p result")

			f = open('./result/%s.sd'%(file_name),'w')
			f.write("filename: %s\n"%(file_name))
			sfw = ''
			for elem in dp.full_data_shape:
				sfw += str(elem) + ' '

			f.write('shape: %s\n'%(sfw))
			f.write('chan: %s\n'%(str(dp.data_contents_memory_shape[0])))
			Vivaldi_memory_dtype = python_dtype_to_Vivaldi_dtype(dp.data_contents_memory_dtype)
			f.write('dtype: %s\n'%(Vivaldi_memory_dtype))
			f.close()






if log_type in ['time','all']:
	for elem in flag_times:
		print "READER",elem, "\t%.3f ms"%(flag_times[elem]*1000)

comm.Barrier()
comm.Disconnect()
