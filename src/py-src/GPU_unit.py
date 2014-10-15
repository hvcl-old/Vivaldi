from mpi4py import MPI

# mpi init
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

def disconnect():
	print "Disconnect", rank, name
	comm.Barrier()
	comm.Disconnect()
	exit()

### GPU_unit description
### GPU unit manage everything related GPU charged
### every data is numpy and it saved in GPU memory

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from Vivaldi_misc import *
from texture import *
from numpy.random import rand
from Vivaldi_memory_packages import Data_package

cuda.init()

data_list = {}
recv_list = {}

gpu_list = []
cpu_list = []
hard_list = []

dirty_flag = {}

mmtx = None

# stream
# memcpy use 1
stream_list = []

global s_requests
s_requests = []

valid_list = []
Event_dict = {}

func_dict = {}
global device_number

# data pools
data_pool = []

numpy.set_printoptions(linewidth=200)

Debug = False

# Debug functions
#################################################################################
def print_devptr(devptr, data_package):
	try:
		data_range = data_package.data_range
		buffer_range = data_package.buffer_range
		buffer_shape = range_to_shape(buffer_range)
		dtype = data_package.data_contents_memory_dtype
		data = numpy.empty(buffer_shape, dtype=dtype)
		cuda.memcpy_dtoh(data, devptr)
		
		m = 10
		dim = len(buffer_range)
		if dim == 1:
			dif_x_l = data_range['x'][0] - buffer_range['x'][0]
			dif_x_r = buffer_range['x'][1] - data_range['x'][1]
			print "PRINT DEVPTR", data[dif_x_l:-dif_x_r]
		if dim == 2:
			print "PRINT DEVPTR", data[m:-m,m:-m]
		if dim == 3:
			print "PRINT DEVPTR", data[m:-m,m:-m,m:-m]
	except:
		print "EXCEPTION When print devptr"
		print data_package

# memory manager
#################################################################################
at = 0
def mem_release(data_package):
	comm.isend(rank,         dest=2,     tag=2)
	comm.isend("release",    dest=2,     tag=2)

	send_data_package(data_package, dest=2, tag=27)

	u = data_package.unique_id
	ss = data_package.split_shape
	sp = data_package.split_position
	data_halo = data_package.data_halo

#################################################################################
global tb_max, tb_cnt, temp_data
tb_max = 0
tb_cnt = 0
temp_data = []

global a
a = numpy.zeros((26),dtype=numpy.int32)

def data_range_to_cuda_in(data_range, full_data_range, buffer_range, data_halo=0, buffer_halo=0, stream=None):
#	global a
	i = 0 
	a = numpy.empty((26), dtype=numpy.int32)
	for axis in AXIS:
		if axis in data_range:
			a[i] = data_range[axis][0]
			a[i+4] = data_range[axis][1]
			a[i+8] = full_data_range[axis][0]
			a[i+12] = full_data_range[axis][1]
			a[i+16] = buffer_range[axis][0]
			a[i+20] = buffer_range[axis][1]
		else:
			a[i] = numpy.int32(0)
			a[i+4] = numpy.int32(0)
			a[i+8] = numpy.int32(0)
			a[i+12] = numpy.int32(0)
			a[i+16] = numpy.int32(0)
			a[i+20] = numpy.int32(0)
			
		i += 1		
	a[24] = numpy.int32(data_halo)
	a[25] = numpy.int32(buffer_halo)
	
	return cuda.In(a)

# GPU short cut function with time measuer

def to_device_with_time(data, data_name, u):
	st = time.time()

	devptr, usage = malloc_with_swap_out(data.nbytes)
	cuda.memcpy_htod(devptr, data)
		
	t = time.time()-st
	ms = 1000*t
	bw = data.nbytes/GIGA/t
	bytes = data.nbytes
	log("rank%d, \"%s\", u=%d, to GPU%d memory transfer, Bytes: %0.2fMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, data_name, u, device_number, bytes/MEGA, ms, bw), 'time', log_type)
	return devptr

# swap out functions
#####################################################################################################################
def cpu_mem_check():
	f = open('/proc/meminfo','r')
	s = f.read()
	f.close()

	idx = s.find('MemFree')
	s = s[idx:]
	idx1 = s.find(':')
	idx2 = s.find('kB')
	s = s[idx1+1:idx2]

	MemFree = s.strip()
	MemFree = int(MemFree)*1024
	return MemFree

def swap_out_to_CPU(elem):
	# prepare variables
	return_falg = True
	u, ss, sp = elem
	dp = data_list[u][ss][sp]
	bytes = dp.data_bytes

	# now we will swap out, this data to CPU
	# so first we should check CPU has enough free memory

	MemFree = cpu_mem_check()

	if log_type in ['memory']:
		fm,tm = cuda.mem_get_info()
		log_str = "CPU MEM CEHCK Before swap out: %s Free, %s Maximum, %s Want to use"%(print_bytes(MemFree),'-',print_bytes(bytes))
		log(log_str,'memory',log_type)


	if bytes > MemFree:
		# not enough memory for swap out to CPU
		return False
	
	# we have enough memory so we can swap out
	# if other process not malloc during this swap out oeprataion

	try:
		buf = numpy.empty((dp.data_memory_shape), dtype= dp.data_contents_memory_dtype)
	except:
		# we failed memory allocation in the CPU
		return False

	# do the swap out
	#cuda.memcpy_dtoh_async(buf, dp.devptr, stream=stream[1])
	cuda.memcpy_dtoh(buf, dp.devptr)
	ctx.synchronize()

	dp.devptr.free()
	dp.devptr = None
	dp.data = buf
	dp.data_dtype = numpy.ndarray
	dp.memory_type = 'memory'


	gpu_list.remove(elem)
	cpu_list.append(elem)

	if log_type in ['memory']:
		fm,tm = cuda.mem_get_info()
		log_str = "GPU MEM CEHCK After swap out: %s Free, %s Maximum, %s Want to use"%(print_bytes(fm),print_bytes(tm),print_bytes(bytes))
		
		log(log_str,'memory',log_type)


	return True

def swap_out_to_hard_disk(elem):
	# prepare variables
	return_falg = True
	u, ss, sp = elem
	dp = data_list[u][ss][sp]
	bytes = dp.data_bytes

    # now we will swap out, this CPU to hard disk
    # so first we should check hard disk has enough free memory
	file_name = '%d_temp'%(rank)
	os.system('df . > %s'%(file_name))

	f = open(file_name)
	s = f.read()
	f.close()

	ss = s.split()

	# get available byte
	avail = int(ss[10])

	if log_type in ['memory']:
		fm,tm = cuda.mem_get_info()
		log_str = "HARD disk MEM CEHCK Before swap out: %s Free, %s Maximum, %s Want to use"%(print_bytes(avail),'-',print_bytes(bytes))
		log(log_str,'memory',log_type)

	if bytes > avail:
		# we failed make swap file in hard disk
		return False

	# now we have enough hard disk to make swap file
	# temp file name, "temp_data, rank, u, ss, sp"
	file_name = 'temp_data, %s, %s, %s, %s'%(rank, u, ss, sp)
	f = open(file_name,'wb')
	f.write(dp.data)
	f.close()

	dp.data = None
	dp.hard_disk = file_name
	dp.memory_type = 'hard_disk'

	cpu_list.remove(elem)
	hard_list.append(elem)
	
	if log_type in ['memory']:
		fm,tm = cuda.mem_get_info()
		log_str = "CPU MEM CEHCK After swap out: %s Free, %s Maximum, %s Want to use"%(print_bytes(fm),print_bytes(tm),print_bytes(bytes))
		log(log_str,'memory',log_type)

	return True

#global CMT
#CMT = 0

def mem_check_and_malloc(bytes):
	fm,tm = cuda.mem_get_info()

	if log_type in ['memory']:
		log_str = "RANK %d, GPU MEM CEHCK before malloc: %s Free, %s Maximum, %s Want to use"%(rank, print_bytes(fm),print_bytes(tm),print_bytes(bytes))
		log(log_str,'memory',log_type)
		

	# we have enough memory

	if fm < bytes:
		# we don't have enough memory, free data fool
		print "BUFFER POOL"
		size = fm
		for elem in list(data_pool):
			usage = elem['usage']
			devptr = elem['devptr']
			devptr.free()
			print "FREE data", usage
			size += usage
			data_pool.remove(elem)
			if size >= bytes: break
	
		fm,tm = cuda.mem_get_info()

	if fm >= bytes:
		# we have enough memory, just malloc
		afm,tm = cuda.mem_get_info()

		devptr = cuda.mem_alloc(bytes)
		
		bfm,tm = cuda.mem_get_info()

		if log_type in ['memory']:
			fm,tm = cuda.mem_get_info()
			log_str = "RANK %d, GPU MALLOC AFTER: %s Free, %s Maximum, %s Want to use"%(rank, print_bytes(fm),print_bytes(tm),print_bytes(bytes))
			log(log_str, 'memory', log_type)
		return True, devptr


	# we don't have enough memory
	return False, None

def data_pool_append(data_pool, devptr, usage, request=None):
	data_pool.append({'devptr':devptr, 'usage':usage, 'request':request})
	
def find_reusable(bytes):
	min = -1
	selected = None

	for elem in data_pool:
		# usage check
		usage = elem['usage']
		if usage < bytes: continue
		if min < usage and min != -1: continue
		if bytes*2 < usage: continue
		
		# request check
		request = elem['request']
		if request != None:
			status = MPI.Status()
			flag = request.Test(status)
			if flag == False: continue
			
		min = usage
		selected = elem

	if selected != None:

		if min < bytes:
			print min, bytes
			assert(False)

		data_pool.remove(selected)
		devptr = selected['devptr']
	#	print "RECYCLE", min, "for", bytes, "DEVPTR", devptr, len(data_pool), rank

		return True, devptr, min # flag, devptr, data size

	return False, None, 0

# malloc GPU array and swap out existing one
def malloc_with_swap_out(bytes, arg_lock=[]):

	# initialize data
	devptr = False

	# check is there reusable data, because malloc and free is expensive operator
	flag, devptr, usage = find_reusable(bytes)

	if flag:
		# reuse empty data
		return devptr, usage

	flag, devptr = mem_check_and_malloc(bytes)
	if flag:
		# we have enough memory
	#	print "MALLOC", bytes, rank
#		print gpu_list
		return devptr, bytes
	else:
		print data_pool


	assert(False)
	# we don't have enough memory, we will try swap out one by one
	for elem in gpu_list:	
		# check we can free or not 
		if elem in arg_lock:
			# this argus should not be swap out, pass 
			continue

	
		while True:
			# swap out to CPU memory, first 
			# if impossible, swap out to hard dsik
			flag1 = swap_out_to_CPU(elem)
			if flag1:
				# swap out succeed
				flag, devptr = mem_check_and_malloc(bytes)
				if flag:
					# malloc succeed
					return devptr

			if flag == False:
				# swap out to CPU failed
				# we need swap out to hard disk
				for elem2 in cpu_list:
					flag1 = swap_out_to_hard_disk(elem2)
					flag2 = swap_out_to_CPU_memory(elem)

					if flag2:
						# swap out to CPU succed
						flag, devptr = mem_check_and_malloc(bytes)
						if flag:
							# malloc succeed
							return devptr
				
				# swap out to hard disk failed
				print "------------------------------------"
				print "VIVALDI ERROR: SWAP OUT to hard dsik failed, may be you don't have enough hard disk sotrage"
				print "WE TRIED WRITE %s to HARD DSIK"%(print_bytes(bytes))
				print "------------------------------------"
				assert(False)

	print "------------------------------------"
	print "VIVALDI ERROR: we shouldn't not be here something wrong"
	print "swap out failed"
	print "------------------------------------"
	assert(False)
	return devptr

def send_data_package(data_package, dest=None, tag=None):
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
	send_data_package(data_package, dest=2, tag=21)

def data_finder(u, ss, sp, gpu_direct=True):

	data_package = data_list[u][ss][sp]
	dp = data_package.copy()

	memory_type = dp.memory_type
	if memory_type == 'devptr':
		if gpu_direct:
			devptr = data_list[u][ss][sp].devptr
			return devptr, dp
		else:
			devptr = data_list[u][ss][sp].devptr
			shape = dp.data_memory_shape
			bcmd = dp.data_contents_memory_dtype
			if log_type in ['time','all']: st = time.time()

			buf = numpy.empty((shape), dtype=bcmd)
			cuda.memcpy_dtoh_async(buf, devptr, stream=stream[1])
#			buf = cuda.from_device(devptr, shape, bcmd)
			if log_type in ['time','all']:
				u = dp.unique_id
				bytes = dp.data_bytes
				t = MPI.Wtime()-st
				ms = 1000*t
				bw = bytes/GIGA/t
				log("rank%d, \"%s\", u=%d, GPU%d data transfer from GPU memory to CPU memory, Bytes: %dMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, name, u, device_number, bytes/MEGA, ms, bw),'time', log_type)
			

			dp.memory_type = 'memory'
			dp.data_dtype = type(buf)
			return buf, dp
	else:
		data = data_list[u][ss][sp].data
		return data, dp
	return None, None
	
def send(data, data_package, dest=None, gpu_direct=True):
	global s_requests
	tag = 52
	dp = data_package
	# send data_package
	send_data_package(dp, dest=dest, tag=tag)

	bytes = dp.data_bytes
	memory_type = dp.memory_type
	
	if log_type in ['time','all']: st = time.time()

	flag = False
	request = None
	if memory_type: # data in the GPU
		if gpu_direct: # want to use GPU direct
			devptr = data
			buf = MPI.make_buffer(devptr.__int__(), bytes)
			request = comm.Isend([buf, MPI.BYTE], dest=dest, tag=57)
			if VIVALDI_BLOCKING: MPI.Request.Wait(request)
			s_requests.append((request, buf, devptr))
			flag = True
		else:# not want to use GPU direct
		
			# copy to CPU
			shape = dp.data_memory_shape
			dtype = dp.data_contents_memory_dtype
			buf = numpy.empty(shape, dtype=dtype)
			cuda.memcpy_dtoh_async(buf, data, stream=stream_list[1])

			request = comm.Isend(buf, dest=dest, tag=57)
			if VIVALDI_BLOCKING: MPI.Request.Wait(request)
			s_requests.append((request, buf, None))
			
	else: # data in the CPU
	
		# want to use GPU direct, not exist case
		# not want to use GPU direct
		if dp.data_dtype == numpy.ndarray: 
			request = comm.Isend(data, dest=dest, tag=57)
			if VIVALDI_BLOCKING: MPI.Request.Wait(request)
			s_requests.append((request, data, None))
			
	if log_type in ['time','all']:
		u = dp.unique_id
		bytes = dp.data_bytes
		t = MPI.Wtime()-st
		ms = 1000*t
		bw = bytes/GIGA/t
	
		if flag:
			log("rank%d, \"%s\", u=%d, from rank%d to rank%d GPU direct send, Bytes: %dMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, name, u, rank, dest, bytes/MEGA, ms, bw),'time', log_type)
		else:
			log("rank%d, \"%s\", u=%d, from rank%d to rank%d MPI data transfer, Bytes: %dMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, name, u, rank, dest, bytes/MEGA, ms, bw),'time', log_type)
	
	return request
	
def recv():
	# DEBUG flag
	################################################
	RECV_CHECK = False
	
	# Implementation
	################################################
	data_package = comm.recv(source=source,	tag=52)
	dp = data_package
	memory_type = dp.memory_type

	if memory_type == 'devptr':
		bytes = dp.data_bytes
		devptr, usage = malloc_with_swap_out(bytes)
		buf = MPI.make_buffer(devptr.__int__(), bytes)
		request = comm.Irecv([buf, MPI.BYTE], source=source, tag=57)

		if VIVALDI_BLOCKING: MPI.Request.Wait(request)

		return devptr, data_package, request, buf
	else:
		data_dtype = dp.data_dtype
		if data_dtype == numpy.ndarray:
			data_memory_shape = dp.data_memory_shape
			dtype = dp.data_contents_memory_dtype
			data = numpy.empty(data_memory_shape, dtype=dtype)

			request = comm.Irecv(data, source=source, tag=57)
			
			if RECV_CHECK: # recv check
				MPI.Request.Wait(request)
				print "RECV CHECK", data
			if VIVALDI_BLOCKING: MPI.Request.Wait(request)

		return data, data_package, request, None

	return None,None,None,None

def save_data(data, data_package):
	# if data is numpy.ndarray, copy to GPU and save only devptr
	dp = data_package

	data_name = dp.data_name
	data_range = dp.data_range
	shape = dp.data_shape
	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
#	log("rank%d, \"%s\", u=%d, saved"%(rank, data_name, u),'general',log_type)
	buf_dtype = type(data)

	if buf_dtype == numpy.ndarray:
		dp.memory_type = 'devptr'
		dp.data_dtype = 'cuda_memory'
		dp.devptr = to_device_with_time(data, data_name, u)

	elif buf_dtype == cuda.DeviceAllocation:
		dp.memory_type = 'devptr'
		dp.data_dtype = 'cuda_memory'
		dp.devptr = data

	elif buf_dtype == 'cuda_memory':
		pass
	else:
		assert(False)

	if dp.devptr == None:
		print dp
		assert(False)
	target = (u,ss,sp)
	if target not in gpu_list:
		gpu_list.append(target)

	# save image
	if log_type in ['image', 'all']:
		if len(dp.data_shape) == 2 and dp.memory_type == 'devptr':
			image_data_shape = dp.data_memory_shape
			md = dp.data_contents_memory_dtype
			a = numpy.empty(image_data_shape,dtype=md)
#			cuda.memcpy_dtoh_async(a, dp.devptr, stream=stream[1])
			cuda.memcpy_dtoh(a, dp.devptr)
			ctx.synchronize()

			buf = a
			extension = 'png'
			dtype = dp.data_contents_dtype
			chan = dp.data_contents_memory_shape
			buf = buf.astype(numpy.uint8)
				
			if chan == [1]: img = Image.fromarray(buf, 'L')
			elif chan == [3]: img = Image.fromarray(buf, 'RGB')
			elif chan == [4]: img = Image.fromarray(buf, 'RGBA')
					
			e = os.system("mkdir -p result")
			img.save('./result/%s%s%s.png'%(u,ss,sp), format=extension)


	u = dp.unique_id
	if u not in data_list: data_list[u] = {}
	if ss not in data_list[u]:data_list[u][ss] = {}
	data_list[u][ss][sp] = dp

dummy = None
def create_helper_textures():
	"create lookup textures for cubic interpolation and random generation"

	def hg(a):
		a2 = a * a
		a3 = a2 * a
		w0 = (-a3 + 3*a2 - 3*a + 1) / 6
		w1 = (3*a3 - 6*a2 + 4) / 6
		w2 = (-3*a3 + 3*a2 + 3*a + 1) / 6
		w3 = a3 / 6
		g = w2 + w3
		h0 = 1 - w1 / (w0 + w1) + a
		h1 = 1 + w3 / (w2 + w3) - a
		return h0, h1, g, 0

	def dhg(a):
		a2 = a * a
		w0 = (-a2 + 2*a - 1) / 2
		w1 = (3*a2 - 4*a) / 2
		w2 = (-3*a2 + 2*a + 1) / 2
		w3 = a2 / 2
		g = w2 + w3
		h0 = 1 - w1 / (w0 + w1) + a
		h1 = 1 + w3 / (w2 + w3) - a
		return h0, h1, g, 0

	tmp = numpy.zeros((256, 4), dtype=numpy.float32)
	for i, x in enumerate(numpy.linspace(0, 1, 256)):
		tmp[i, :] = numpy.array(hg(x))

	tmp = numpy.reshape(tmp, (1, 256, 4))

	hg_texture = create_2d_rgba_texture(tmp, mod, 'hgTexture')
	tmp = numpy.zeros((256, 4), dtype=numpy.float32)
	for i, x in enumerate(numpy.linspace(0, 1, 256)):
		tmp[i, :] = numpy.array(dhg(x))
	tmp = numpy.reshape(tmp, (1, 256, 4))
	dhg_texture = create_2d_rgba_texture(tmp, mod, 'dhgTexture')

	tmp = numpy.zeros((256, 256), dtype=numpy.uint32) # uint32(rand(256, 256) * (2 << 30))
	random_texture = create_2d_texture(tmp, mod, 'randomTexture', True)
	update_random_texture(random_texture)

	# to prevent GC from destroying the textures
	global dummy
	dummy = (hg_texture, dhg_texture)

def update_random_texture(random_texture):
	tmp = numpy.uint32(rand(256, 256) * (2 << 30))
	update_2d_texture(random_texture, tmp)

global KD
KD = []

def kernel_write(function_name, dest_devptr, dest_info, source_devptr, source_info, work_range, stream=None):
	global KD

	# initialize variables
	global tb_cnt
	tb_cnt = 0

	# dest
	cuda_args = [dest_devptr]
	cuda_args += [dest_info]

	# source
	cuda_args += [source_devptr]
	cuda_args += [source_info]

	# work_range
	cuda_args += make_cuda_list(work_range)

	# initialize model view
	eye = numpy.eye(4,dtype=numpy.float32)
	cuda.memcpy_htod_async(mmtx, eye, stream=stream)
	cuda.memcpy_htod_async(inv_mmtx, eye, stream=stream)
	
	try:
		if Debug:
			print "Function name: ", function_name
		func = mod.get_function(function_name) #cutting function
	except:
		print "Function not found ERROR"
		print "Function name: ", function_name
		assert(False)
		
	# set work range
	block, grid = range_to_block_grid(work_range)

	if log_type in ['time', 'all']:
		st = time.time()

	func(*cuda_args, block=block, grid=grid, stream=stream)

	#ctx.synchronize()
	
	KD.append((dest_info, source_info))
	

	if log_type in ['time', 'all']:
		bytes = make_bytes(work_range,3)
		t = MPI.Wtime()-st
		ms = 1000*t
		bw = bytes/GIGA/t
		log("rank%d, GPU%d, , kernel write time, Bytes: %dMB, time: %.3f ms, speed: %.3f GByte/sec "%(rank, device_number, bytes/MEGA, ms, bw),'time', log_type)


		
def wait_data_arrive(data_package, stream=None):
	# local_functions
	#################################################################
	def prepare_dest_package_and_dest_devptr(new_data, dest_package):
		output_package = dest_package.copy()
		if new_data:
			# create new data
			dest_devptr, new_usage = malloc_with_swap_out(output_package.buffer_bytes)
			output_package.usage = new_usage
			cuda.memset_d8(dest_devptr, 0, output_package.buffer_bytes)
		else:
			new_data_halo = task.dest.data_halo
			exist_data_halo = data_list[u][ss][sp]
			
			if new_data_halo < exist_data_halo: output_package = data_list[u][ss][sp]
			else: output_package = dest_package
			
			dest_devptr = data_list[u][ss][sp].devptr

		return output_package, dest_devptr
	def new_data_check(data_exist, dest_devptr):
		new_data = False
		if not data_exist: # data is not exist
			new_data = True
		else:
			# buffer not exist
			if dest_devptr == None: 
				new_data = True
		return new_data
	
	# initialize variables
	#################################################################
	u = data_package.unique_id
	ss = data_package.split_shape
	sp = data_package.split_position

	# buffer exist flag
	flag = True
	if u not in recv_list: flag = False
	elif ss not in recv_list[u]: flag = False
	elif sp not in recv_list[u][ss]: flag = False

	# implementation
	#####################################################################
	if flag:
		# initialize variable
		func_name = 'writing'
		first = recv_list[u][ss][sp][0]
		dt = first['task'].source.data_contents_dtype 
		di = len(first['task'].source.data_shape)
		func_name += '_%dd_%s'%(di, dt)
		
		# prepare copy
		# make dest data
		dest_package = first['task'].dest

		# check data exist
		dest_devptr = None
		data_exist = False
		usage = 0
		if u in data_list:
			if ss in data_list[u]:
				if sp in data_list[u][ss]:
					dest_devptr = data_list[u][ss][sp].devptr
					usage = data_list[u][ss][sp].buffer_bytes
					data_exist = True

		# data exist check
		new_data = new_data_check(data_exist, dest_devptr)
		
		# prepare dest_package and dest_devptr		
		dest_package, dest_devptr = prepare_dest_package_and_dest_devptr(new_data, dest_package)
		
		dest_data_range = dest_package.data_range
		dest_buffer_range = dest_package.buffer_range
	
		# write from temp data to dest data
		for elem in recv_list[u][ss][sp]:
			# wait data arrive
			if 'request' in elem: 
				MPI.Request.Wait(elem['request'])
				
			# memory copy have release
			if 'mem_release' in elem:
				mem_release(elem['task'].source)
				elem['t_devptr'] = elem['data']
				# we don't need rewrite again
				if data_exist and new_data == False:
					# data already existed and keep using same data
					continue

			t_data = elem['data']
			t_dp = elem['data_package']
			work_range = elem['task'].work_range
			t_data_range = t_dp.data_range
			t_buffer_range = t_dp.buffer_range
			t_data_halo = t_dp.data_halo
			t_buffer_halo = t_dp.buffer_halo
			
			if type(t_data) == numpy.ndarray:
				t_devptr, usage = malloc_with_swap_out(t_dp.data_bytes)
				cuda.memcpy_htod_async(t_devptr, t_data, stream=stream)
				elem['t_devptr'] = t_devptr
				elem['usage'] = usage	
				
			else:
				t_devptr = t_data
				elem['t_devptr'] = t_devptr
				
			dest_info = data_range_to_cuda_in(dest_data_range, dest_data_range, dest_buffer_range, data_halo=dest_package.data_halo, buffer_halo=dest_package.buffer_halo, stream=stream)
			t_info = data_range_to_cuda_in(t_data_range, t_data_range, t_buffer_range, data_halo=t_data_halo, buffer_halo=t_buffer_halo, stream=stream)
				
			kernel_write(func_name, dest_devptr, dest_info, t_devptr, t_info, work_range, stream=stream)
		
			target = (u,ss,sp)
			if target not in gpu_list:
				gpu_list.append((u,ss,sp))

		#stream.synchronize()
		for elem in recv_list[u][ss][sp]:
			t_data = elem['data']
			t_dp = elem['data_package']
			work_range = elem['task'].work_range
			t_data_range = t_dp.data_range
			t_devptr = elem['t_devptr']
			usage = elem['usage']

			if 'temp_data' in elem:
				data_pool_append(data_pool, t_devptr, usage)

		del recv_list[u][ss][sp]
		if recv_list[u][ss] == {}: del recv_list[u][ss]
		if recv_list[u] == {}: del recv_list[u]

		dest_package.memory_type = 'devptr'
		dest_package.data_dtype = 'cuda_memory'
		dest_package.devptr = dest_devptr

		if not data_exist:
			if u not in data_list: data_list[u] = {}
			if ss not in data_list[u]: data_list[u][ss] = {}
		#print "RESULT", dest_package
		data_list[u][ss][sp] = dest_package
	else:
		# data already in here
		pass

def memcpy_p2p_send(task, dest):
	# initialize variables
	ts = task.source
	u = ts.unique_id
	ss = ts.split_shape
	sp = ts.split_position
	
	wr = task.work_range
	
	dest_package = task.dest
	du = dest_package.unique_id
	dss = dest_package.split_shape
	dsp = dest_package.split_position

	func_name = 'writing'
	dt = ts.data_contents_dtype 
	di = len(ts.data_shape)
	func_name += '_%dd_%s'%(di, dt)

	if du not in dirty_flag: dirty_flag[du] = {}
	if dss not in dirty_flag[du]: dirty_flag[du][dss] = {}
	if dsp not in dirty_flag[du][dss]: dirty_flag[du][dss][dsp] = True

	# wait source is created
	target = (u,ss,sp)
	if target in valid_list:
		Event_dict[target].synchronize()
	else:
		wait_data_arrive(task.source)
		
	# check copy to same rank or somewhere else
	
#	print "p2p_send", task.source.data_range, dest_package.data_range, wr, rank, dest
	sr = (task.execid == rank)
	if sr:
		# source is not necessary anymore
		if du not in recv_list: recv_list[du] = {}
		if dss not in recv_list[du]: recv_list[du][dss] = {}
		if dsp not in recv_list[du][dss]: recv_list[du][dss][dsp] = []

		recv_list[du][dss][dsp].append({'task':task, 'data':data_list[u][ss][sp].devptr, 'data_package':data_list[u][ss][sp],'mem_release':True,'usage':data_list[u][ss][sp].usage})

		if VIVALDI_BLOCKING:
			wait_data_arrive(task.dest)
		notice(task.dest)
	else:
		# different machine
		# check data will be cutted or not
		source_package = data_list[u][ss][sp]

		cut = False
		if wr != source_package.buffer_range: cut = True

		bytes = 0
		data_halo = 0
		
		# make data package
		dp = data_list[u][ss][sp].copy()
		dp.unique_id = u
		dp.split_shape = ss
		dp.split_position = sp
		dp.data_halo = data_halo
		dp.set_data_range(wr)
		
		# prepare data
		if cut:
			dp.set_buffer_range(0)
			
			bcms = source_package.data_contents_memory_shape
			bcmd = source_package.data_contents_memory_dtype
			bcmb = get_bytes(bcmd)

			bytes = make_bytes(wr, bcms, bcmb)
			dest_devptr, usage = malloc_with_swap_out(bytes)
			start = time.time()
			
			dest_info = data_range_to_cuda_in(wr, wr, wr, stream=stream_list[1])
			source_info = data_range_to_cuda_in(source_package.data_range, source_package.full_data_range, source_package.buffer_range, data_halo=source_package.data_halo, buffer_halo=source_package.buffer_halo, stream=stream_list[1])
			
			kernel_write(func_name, dest_devptr, dest_info, source_package.devptr, source_info, wr, stream=stream_list[1])
			
		
			start = time.time()
			data_halo = 0
			temp_data = True
			
		else:
			dest_devptr = source_package.devptr
			bytes = source_package.data_bytes
			data_halo = source_package.data_halo
			usage = source_package.usage
			temp_data = False

		# now dest_devptr has value
	
		# say dest to prepare to recv this data
		comm.isend(rank,				dest=dest,	tag=5)
		comm.isend("memcpy_p2p_recv",	dest=dest,	tag=5)
		comm.isend(task,				dest=dest,	tag=57)
		comm.isend(data_halo,			dest=dest,	tag=57)
		
		# send data
		target = (u,ss,sp)
		if target in Event_dict:
			Event_dict[target].synchronize()
		
		if target not in valid_list:
			wait_data_arrive(dp)
			
		data = dest_devptr				
		stream_list[1].synchronize()
		
		request = send(data, dp, dest=dest, gpu_direct=GPUDIRECT)

		if temp_data:
			data_pool_append(data_pool, dest_devptr, usage, request=request)
			pass

def memcpy_p2p_recv(task, data_halo):
	# initialize variables
	dest_package = task.dest
	du = dest_package.unique_id
	dss = dest_package.split_shape
	dsp = dest_package.split_position

	func_name = 'writing'
	dt = task.source.data_contents_dtype 
	di = len(task.source.data_shape)
	func_name += '_%dd_%s'%(di, dt)

	data, data_package, request, buf = recv()
	dp = data_package

	# data exist check
	if du not in data_list: 
		data_exist = False
		data_list[du] = {}
	if dss not in data_list[du]:
		data_exist = False
		data_list[du][dss] = {}
	if dsp not in data_list[du][dss]:
		data_exist = False
		data_list[du][dss][dsp] = dest_package

	# data already exist
	if du not in recv_list: recv_list[du] = {}
	if dss not in recv_list[du]: recv_list[du][dss] = {}
	if dsp not in recv_list[du][dss]: recv_list[du][dss][dsp] = []

	recv_list[du][dss][dsp].append({'request':request,'task':task, 'data':data, 'data_package':data_package, 'buf': buf, 'temp_data':True,'usage':dp.data_bytes})
	
	notice(dest_package)
	return 

global FD
FD = []

def run_function(function_package):
	# global variables
	global FD
	global tb_cnt
	
	# initialize variables
	fp = function_package
	func_output = fp.output
	u = func_output.unique_id
	ss = func_output.split_shape
	sp = func_output.split_position
	data_halo = func_output.data_halo
	function_name = fp.function_name

	args = fp.function_args
	work_range = fp.work_range

	tb_cnt = 0
	
	stream = stream_list[0]

#	cuda.memcpy_htod_async(bandwidth, numpy.float32(fp.TF_bandwidth), stream=stream)
#	cuda.memcpy_htod_async(front_back, numpy.int32(fp.front_back), stream=stream)
	if fp.update_tf == 1:
		tf.set_filter_mode(cuda.filter_mode.LINEAR)
		cuda.bind_array_to_texref(cuda.make_multichannel_2d_array(fp.trans_tex.reshape(1,256,4), order='C'), tf)
		cuda.memcpy_htod_async(bandwidth, numpy.float32(fp.TF_bandwidth), stream=stream)
	if fp.update_tf2 == 1:
		tf1.set_filter_mode(cuda.filter_mode.LINEAR)
		cuda.bind_array_to_texref(cuda.make_multichannel_2d_array(fp.trans_tex.reshape(1,256,4), order='C'), tf1)
		cuda.memcpy_htod_async(bandwidth, numpy.float32(fp.TF_bandwidth), stream=stream)
	
	cuda_args = []

	data_exist = True
	if u not in data_list: data_exist = False
	elif ss not in data_list[u]: data_exist = False
	elif sp not in data_list[u][ss]: data_exist = False

	if data_exist:
		# initialize variables
		data_package = data_list[u][ss][sp]
		dp = data_package

		if dp.devptr == None:
			wait_data_arrive(data_package, stream=stream)
		###########################
		devptr = dp.devptr
		output_range = dp.data_range
		full_output_range = dp.full_data_range
		ad = data_range_to_cuda_in(output_range, full_output_range, stream=stream)
		cuda_args += [ad]
		output_package = dp
		
		FD.append(ad)
	else:
		bytes = func_output.buffer_bytes
		devptr, usage = malloc_with_swap_out(bytes)
		log("created output data bytes %s"%(str(func_output.buffer_bytes)),'detail',log_type)
		data_range = func_output.data_range
		full_data_range = func_output.full_data_range
		buffer_range = func_output.buffer_range
		buffer_halo = func_output.buffer_halo
		ad = data_range_to_cuda_in(data_range, full_data_range, buffer_range, buffer_halo=buffer_halo, stream=stream)

		cuda_args += [ad]
		output_package = func_output
		output_package.buffer_bytes = usage
		
		if False:
			print "OUTPUT"
			print "OUTPUT_RANGE", data_range
			print "OUTPUT_FULL_RANGE",  full_data_range
			
		FD.append(ad)

	# set work range
	block, grid = range_to_block_grid(work_range)
	# set block and grid
#	log("work_range "+str(work_range),'detail',log_type)
#	log("block %s grid %s"%(str(block),str(grid)),'detail',log_type)

	cuda_args = [devptr] + cuda_args

#	print "GPU", rank, "BEFORE RECV", time.time()
	# Recv data from other process
	for data_package in args:
		u = data_package.unique_id
		data_name = data_package.data_name
		if data_name not in work_range and u != -1:
			wait_data_arrive(data_package, stream=stream)

#	print "GPU", rank, "Recv Done", time.time()
	# set cuda arguments 
	for data_package in args:
		data_name = data_package.data_name
		data_dtype = data_package.data_dtype
		data_contents_dtype = data_package.data_contents_dtype

		u = data_package.unique_id

		if data_name in work_range:
			cuda_args.append( numpy.int32(work_range[data_name][0]))
			cuda_args.append( numpy.int32(work_range[data_name][1]))

		elif u == -1:
			data = data_package.data
			dtype = type(data)

			if dtype in [int]: data = numpy.float32(data)
			if dtype in [float]: data = numpy.float32(data)

			cuda_args.append(numpy.float32(data)) # temp
		else:
			ss = data_package.split_shape
			sp = data_package.split_position
			dp = data_list[u][ss][sp] # it must be fixed to data_package latter

			memory_type = dp.memory_type
			if memory_type == 'devptr':

				cuda_args.append(dp.devptr)
				data_range = dp.data_range
				full_data_range = dp.full_data_range
				buffer_range = dp.buffer_range
				
				if False:
					print "DATA_NAME", data_name
					print "DATA_RANGE", data_range
					print "FULL_DATA_RANGE", full_data_range
					print "BUFFER_RANGE", buffer_range
					print "DATA_HALO", dp.data_halo
					print "BUFFER_HALO", dp.buffer_halo
					print dp
	
					print_devptr(dp.devptr, dp)
				ad = data_range_to_cuda_in(data_range, full_data_range, buffer_range, data_halo=dp.data_halo, buffer_halo=dp.buffer_halo, stream=stream)

				cuda_args.append(ad)
				FD.append(ad)
		
#	log("function cuda name %s"%(function_name),'detail',log_type)

#	if function_name in func_dict:
#		func = func_dict[function_name]
#	else:


	# set modelview matrix
	cuda.memcpy_htod_async(mmtx, fp.mmtx.reshape(16), stream=stream)
	cuda.memcpy_htod_async(inv_mmtx, fp.inv_mmtx.reshape(16), stream=stream)

	try:
		if Debug:
			print "Function name: ", function_name
		func = mod.get_function(function_name.strip())
	except:
		print "Function not found ERROR"
		print "Function name: " + function_name
		assert(False)
	
	stream_list[0].synchronize()

	if log_type in ['time','all']:
		start = time.time()

	kernel_finish = cuda.Event()
	func( *cuda_args, block=block, grid=grid, stream=stream_list[0])
	kernel_finish.record(stream=stream_list[0])

	"""
	try:
		a = numpy.empty((30,30),dtype=numpy.int32)
		cuda.memcpy_dtoh(a, cuda_args[0])
		print a[10:-10,10:-10]
	except:
		print "Fail", function_name
		print "Fp.output", fp.output
		pass
	"""
	
	u = func_output.unique_id
	ss = func_output.split_shape
	sp = func_output.split_position

	target = (u,ss,sp)

	Event_dict[target] = kernel_finish
	if target not in valid_list:
		valid_list.append(target)

	#################################################################################
	# finish
	if log_type in ['time','all']:
		t = (time.time() - start)
		ms = 1000*t

		log("rank%d, %s,  \"%s\", u=%d, GPU%d function running,,, time: %.3f ms "%(rank, func_output.data_name, function_name, u, device_number,  ms),'time',log_type)
	#log("rank%d, \"%s\", GPU%d function finish "%(rank, function_name, device_number),'general',log_type)

	###################################################################################
	# decrease retain_count
	for data_package in args:
		u = data_package.unique_id
		if u != -1:
			mem_release(data_package)

#	print "Release", time.time()
	return devptr, output_package

# main
##############################################################################################################################
if '-L' in sys.argv:
	idx = sys.argv.index('-L')
	log_type = sys.argv[idx+1]

GPUDIRECT = True
if '-G' in sys.argv:
	idx = sys.argv.index('-G')
	GPUDIRECT = sys.argv[idx+1]
	if GPUDIRECT == 'on':GPUDIRECT = True
	else: GPUDIRECT = False
		
VIVALDI_BLOCKING = False
if '-B' in sys.argv:
	idx = sys.argv.index('-B')
	VIVALDI_BLOCKING = sys.argv[idx+1]
	if VIVALDI_BLOCKING.lower() in ['true','on']: VIVALDI_BLOCKING = True
	elif VIVALDI_BLOCKING.lower() in ['false','off']: VIVALDI_BLOCKING = False

global device_number
device_number = rank-int(sys.argv[1])
log("rank%d, GPU unit launch at %s and device_numver is %d"%(rank, name, device_number),'general',log_type)

try:
	dev = cuda.Device(device_number)
	ctx = dev.make_context()
#	comm.isend("device_available", dest=0, tag=777)
	comm.send("device_available", dest=0, tag=777)
except:
	log("rank%d, Fail to start GPU%d"%(rank, device_number),'general',log_type)
#	comm.isend("device_unavailable", dest=0, tag=777)
	comm.send("device_unavailable", dest=0, tag=777)
	comm.Barrier()
	comm.Disconnect()
	exit()

n = 2
flag = ''
GPU_TIME = 0

for elem in range(2):
	stream_list.append(cuda.Stream())

flag_times = {}
for elem in ["recv","send_order","free","memcpy_p2p_send","memcpy_p2p_recv","request_data","data_package","deploy_func","run_function","wait"]:
	flag_times[elem] = 0


# get function problem main
################################################################################################################

# main loop
################################################################################################################
while flag != "finish":
	#print "GPU ", rank, "WAIT", time.time()
	
	st = time.time()
	
	log("rank%d, GPU%d Waiting "%(rank, device_number),'general',log_type)
	source = comm.recv(source=MPI.ANY_SOURCE, tag=5)
	flag = comm.recv(source=source, tag=5)
	flag_times['wait'] += time.time() - st

	log("rank%d, GPU%d flag: %s"%(rank, device_number, flag),'general',log_type)
	if flag == "synchronize":
		# synchronize

		# requests wait
		f_requests = []
		for u in recv_list:
			for ss in recv_list[u]:
				for sp in recv_list[u][ss]:
					f_requests += [elem['request'] for elem in recv_list[u][ss][sp]]

		f_requests += [elem[0] for elem in s_requests]

#		print "BF", rank
		MPI.Request.Waitall(f_requests)

		# stream synchronize
		for stream in stream_list: stream.synchronize()

		# context synchronize
		ctx.synchronize()
		comm.send("Done", dest=source, tag=999)

	elif flag == "recv":
		st = time.time()
		data, data_package, request = recv()
		dp = data_package
		notice(dp)
		save_data(data, dp)
		idle()
		flag_times[flag] += time.time() - st

	elif flag == "send_order":
		st = time.time()
		dest = comm.recv(source=source, tag=53)
		u = comm.recv(source=source, tag=53)
		ss = comm.recv(source=source, tag=53)
		sp = comm.recv(source=source, tag=53)

		log("rank%d, order to send data u=%d to %d"%(rank, u, dest),'general',log_type)

		data, data_package = data_finder(u,ss,sp, gpu_direct)
		dp = data_package
		ctx.synchronize()
		send(data, dp, dest=dest, gpu_direct=gpu_direct)
		idle()
		flag_times[flag] += time.time() - st

	elif flag == "free":
		st = time.time()
		u = comm.recv(source=source, tag=58)
		ss = comm.recv(source=source, tag=58)
		sp = comm.recv(source=source, tag=58)
		data_halo = comm.recv(source=source, tag=58)

		# check this data is really exsit
		if u not in data_list: continue
		if ss not in data_list[u]: continue
		if sp not in data_list[u][ss]: continue

		# data exist
		if log_type in ['memory']:
			log_str = "RANK%d, FREE BF "%(rank) + str(cuda.mem_get_info()) + " %s %s %s"%(u, ss, sp)
			log(log_str, 'memory', log_type)
		
		target = (u,ss,sp)

		if target in gpu_list:
			gpu_list.remove(target)
#			data_list[u][ss][sp].devptr.free()

#			if data_list[u][ss][sp].usage == 0:
#				print data_list[u][ss][sp]
#				assert(False)
			data_pool_append(data_pool, data_list[u][ss][sp].devptr, data_list[u][ss][sp].usage)
#			print "APPEND C", len(data_pool), rank
#			print data_list[u][ss][sp].data_bytes, data_list[u][ss][sp].usage

			del(data_list[u][ss][sp])
			if data_list[u][ss] == {}: del(data_list[u][ss])
			if data_list[u] == {}: del(data_list[u])
		else:
			assert(False)

		if target in cpu_list: cpu_list.remove(target)
		if target in hard_list: hard_list.remove(target)

		if log_type in ['memory']:
			log_str = "RANK%d, FREE AF "%(rank)+ str(cuda.mem_get_info())+ " %s %s %s"%(u, ss, sp)
			log(log_str, 'memory', log_type)

		flag_times[flag] += time.time() - st

	elif flag == "memcpy_p2p_send":
		st = time.time()
		dest = comm.recv(source=source, tag=56)
		task = comm.recv(source=source, tag=56)

		memcpy_p2p_send(task, dest)
		idle()
		flag_times[flag] += time.time() - st

	elif flag == "memcpy_p2p_recv":
		st = time.time()
		task = comm.recv(source=source, tag=57)
		data_halo = comm.recv(source=source, tag=57)
		memcpy_p2p_recv(task, data_halo)

		mem_release(task.source)
		idle()
		flag_times[flag] += time.time() - st

	elif flag == "request_data":
		st = time.time()
		u = comm.recv(source=source, tag=55)
		ss = comm.recv(source=source, tag=55)
		sp = comm.recv(source=source, tag=55)
		log("rank%d, send data u=%d to %d"%(rank, u, source),'general',log_type)
		gpu_direct = comm.recv(source=source, tag=52)
		data, data_package = data_finder(u,ss,sp, gpu_direct)
		send(data, data_package, dest=source, gpu_direct=gpu_direct)
		flag_times[flag] += time.time() - st

	elif flag == "deploy_func":
		st = time.time()
		
		code = comm.recv(source=source, tag=54)

		log("rank%d, device function is deployed"%(rank),'general',log_type)
		write_file('asdf.cu', code)	

		mod = SourceModule(code, no_extern_c = True, options = ["-use_fast_math", "-O3"])

		temp,_ = mod.get_global('DEVICE_NUMBER')
		cuda.memcpy_htod(temp, numpy.int32(device_number))
		mmtx,_ = mod.get_global('modelview')
		inv_mmtx,_ = mod.get_global('inv_modelview')
		bandwidth,_ = mod.get_global('TF_bandwidth')
		tf = mod.get_texref('TFF')
		tf1 = mod.get_texref('TFF1')
	
		create_helper_textures()
		log("rank%d, device function now available"%(rank),'general',log_type)
		flag_times[flag] += time.time() - st

		ctx.synchronize()
		fm,tm = cuda.mem_get_info()
#		print "START GPU MEMORY", fm , tm
		free_mem = fm

	elif flag == "run_function":
		st = time.time()
		function_package = comm.recv(source=source, tag=51)
		fp = function_package
		GG = time.time()
		devptr, output_package = run_function(fp)

		notice(fp.output)
		save_data(devptr, output_package)
		idle()

		flag_times[flag] += time.time() - st

	else:
		pass


if log_type in ['time','all']:
	for elem in flag_times:
		print "GPU ",elem, "\t%.3f ms"%(flag_times[elem]*1000)


#fm, tm = cuda.mem_get_info()

if False:
	print "RANK", rank
	print "END MEMORY", fm, tm
	print "END - START DIFF", free_mem - fm
	print "GPU_LIST", gpu_list
comm.Barrier()
comm.Disconnect()

del(data_list)
del(dirty_flag)
ctx.pop()



del(temp)
del(mmtx)
del(inv_mmtx)
del(bandwidth)
del(mod)


MPI.Finalize()
