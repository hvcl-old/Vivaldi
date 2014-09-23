from mpi4py import MPI

# mpi init
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

	
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from Vivaldi_misc import *
from texture import *
from numpy.random import rand
from Vivaldi_memory_packages import Data_package
import traceback

def disconnect():
	print "Disconnect", rank, name
	comm.Disconnect()
	MPI.Finalize()
	exit()

# MPI Communication
def wait_data_arrive(data_package, stream=None):
	# local_functions
	#################################################################
	def prepare_dest_package_and_dest_devptr(new_data, dest_package):
		# ???????????????????
		output_package = dest_package.copy()
		if new_data:
			# create new data
			# because we don't have cuda memory allocation for dest_package
			dest_devptr, new_usage = malloc_with_swap_out(output_package.data_bytes)
			output_package.set_usage(new_usage)
			
			cuda.memset_d8(dest_devptr, 0, output_package.data_bytes)
		else:
			# we already have cuda memory allocation
			# if there are enough halo, we can use exist buffer instead dest_package
			# if there are not enough halo, we have to allocate new buffer
			
			new_data_halo = task.dest.data_halo
			exist_data_halo = data_list[u][ss][sp].data_halo
			
			if new_data_halo <= exist_data_halo: 
				output_package = data_list[u][ss][sp]
			else:
				output_package = dest_package
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
	u, ss, sp = data_package.get_id()
	
	# check recv_list
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
					usage = data_list[u][ss][sp].data_bytes
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
		data_list[u][ss][sp] = dest_package
	else:
		# data already in here
		pass
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
def memcpy_p2p_send(task, dest):
	# initialize variables
	ts = task.source
	u, ss, sp = ts.get_id()
	
	wr = task.work_range
	
	dest_package = task.dest
	du, dss, dsp = dest_package.get_id()

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
def memcpy_p2p_recv(task, data_halo):
	# initialize variables
	dest_package = task.dest
	du, dss, dsp = dest_package.get_id()
	
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
def mem_release(data_package):
	comm.isend(rank,                dest=1,    tag=5)
	comm.isend("release",           dest=1,    tag=5)
	send_data_package(data_package, dest=1,    tag=57)
	
# Communication with scheduler
def send_data_package(data_package, dest=None, tag=None):
	dp = data_package
	t_data, t_devptr = dp.data, dp.devptr
	dp.data, dp.devptr = None, None
	comm.isend(dp, dest=dest, tag=tag)
	dp.data, dp.devptr = t_data, t_devptr
	t_data,t_devptr = None, None
def idle():
	rank = comm.Get_rank()
	comm.isend(rank,                dest=1,    tag=5)
	comm.isend("idle",              dest=1,    tag=5)
def notice(data_package):
	rank = comm.Get_rank()
	comm.isend(rank,                dest=1,    tag=5)
	comm.isend("notice",            dest=1,    tag=5)
	send_data_package(data_package, dest=1,    tag=51)
	
# local functions
def get_function_dict(x):
	function_code_dict = {}
	def get_function_name_list(code=''):
		def get_head(code='', st=0):
			idx = code.find('def ', st)
			if idx == -1: return None, -1
			idx2 = code.find(':', idx+1)
			head = code[idx+4:idx2]
			head = head.strip()
			return head, idx2+1	
		def get_function_name(head=''):
			idx = head.find('(')
			return head[:idx]
			
		function_name_list = []
		st = 0
		while True:
			head, st = get_head(code, st)
			if head == None: break
			function_name = get_function_name(head)
			function_name_list.append(function_name)
		return function_name_list	
	def get_code(function_name='', code=''):
		if function_name == '':
			print "No function name found"
			assert(False)
		# initialization
		###################################################
		st = 'def '+function_name+'('
		output = ''
		s_idx = code.find(st)
		
		if s_idx == -1:
			print "Error"
			print "Cannot find the function"
			print "Function want to find:", function_name
			assert(False)
		n = len(code)
		
		
		# implementation
		###################################################
		
		# there are n case function finish
		# ex1) code end
		# def main()
		# 	...
		#
		# ex2) another function 
		# def main()
		# 	...
		# def func():
		#
		# ex3) indent
		# def main()
		# 	...
		# print 
		
		# there are n case main not finish
		# ex1) main
		# def main():
		#     ArithmeticError
		#
		#     BaseException
		
		def get_indent(line): 
			s_line = line.strip()
			i_idx = line.find(s_line)
			indent = line[:i_idx]

			return indent
			
		line = ''
		cnt = 1
		i = s_idx
		while i < n:
			w = code[i]
			
			line += w
			if w == '\n':
				indent = get_indent(line)
				if indent == '' and line.strip().startswith('def'):
					# ex2 and ex3
					if cnt == 0:break
					cnt -= 1
				
				output += line
				line = ''
			i += 1
		
		return output
		
	function_name_list = get_function_name_list(x)
	for function_name in function_name_list:
		function_code = get_code(function_name=function_name, code=x)
		function_code_dict[function_name] = function_code
	return function_code_dict
def save_data(data, data_package):
	# if data is numpy.ndarray, copy to GPU and save only devptr
	dp = data_package

	data_name = dp.data_name
	data_range = dp.data_range
	shape = dp.data_shape
	u, ss, sp = dp.get_id()
	
	#log("rank%d, \"%s\", u=%d, saved"%(rank, data_name, u),'general',log_type)
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


	u = dp.get_unique_id()
	if u not in data_list: data_list[u] = {}
	if ss not in data_list[u]:data_list[u][ss] = {}
	data_list[u][ss][sp] = dp
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
def to_device_with_time(data, data_name, u):
	st = time.time()

	devptr, usage = malloc_with_swap_out(data.nbytes)
	cuda.memcpy_htod(devptr, data)
		
	t = time.time()-st
	ms = 1000*t
	bw = data.nbytes/GIGA/t
	bytes = data.nbytes
	#log("rank%d, \"%s\", u=%d, to GPU%d memory transfer, Bytes: %0.2fMB, time: %.3f ms, speed: %.3f GByte/sec"%(rank, data_name, u, device_number, bytes/MEGA, ms, bw), 'time', log_type)
	return devptr
# CUDA support functions
def print_devptr(devptr, data_package):
	try:
		data_range = data_package.data_range
		dtype = data_package.data_contents_memory_dtype
		data_memory_shape = data_package.data_shape
		data = numpy.empty(data_memory_shape, dtype=dtype)
		cuda.memcpy_dtoh(data, devptr)
		
		print "PRINT DEVPTR", data
	except:
		print data_package.info()
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

	mod = source_module_dict['default']
	
	func = mod.get_function(function_name)
	
	# set work range
	block, grid = range_to_block_grid(work_range)
	
	if log_type in ['time', 'all']:
		st = time.time()

#	print "BL", block, grid
	func(*cuda_args, block=block, grid=grid, stream=stream)
	ctx.synchronize()
	
	KD.append((dest_info, source_info))
	

	if log_type in ['time', 'all']:
		bytes = make_bytes(work_range,3)
		t = MPI.Wtime()-st
		ms = 1000*t
		bw = bytes/GIGA/t
		log("rank%d, GPU%d, , kernel write time, Bytes: %dMB, time: %.3f ms, speed: %.3f GByte/sec "%(rank, device_number, bytes/MEGA, ms, bw),'time', log_type)
	
# CUDA compile functions
def compile_for_GPU(function_package, kernel_function_name='default'):
	kernel_code = ''
	if kernel_function_name == 'default':
		kernel_code = attachment
		source_module_dict[kernel_function_name] = SourceModule(kernel_code, no_extern_c = True, options = ["-use_fast_math", "-O3"])
	else:
		fp = function_package
		
		from vivaldi_translator import translate_to_CUDA
		function_name = fp.function_name

		Vivaldi_code = function_code_dict[function_name]
		
		function_code = translate_to_CUDA(Vivaldi_code=Vivaldi_code, function_name=function_name, function_arguments=fp.function_args)
		
		kernel_code = attachment + 'extern "C"{\n'
		kernel_code += function_code
		kernel_code += '\n}'
		#print function_code
		source_module_dict[kernel_function_name] = SourceModule(kernel_code, no_extern_c = True, options = ["-use_fast_math", "-O3"])

		temp,_ = source_module_dict[kernel_function_name].get_global('DEVICE_NUMBER')
		cuda.memcpy_htod(temp, numpy.int32(device_number))
		
		func_dict[kernel_function_name] = source_module_dict[kernel_function_name].get_function(kernel_function_name)
		
		create_helper_textures(source_module_dict[kernel_function_name])
dummy = None
def create_helper_textures(mod):
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

# CUDA malloc function
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

# CUDA execute function
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
			# i'm thinking about not use buffer range any more
			# but we can't change GPU immediately
			#a[i+16] = buffer_range[axis][0] 
			#a[i+16] = buffer_range[axis][0]
			a[i+16] = data_range[axis][0]
			a[i+20] = data_range[axis][1]
		else:
			a[i] = numpy.int32(0)
			a[i+4] = numpy.int32(0)
			a[i+8] = numpy.int32(0)
			a[i+12] = numpy.int32(0)
			a[i+16] = numpy.int32(0)
			a[i+20] = numpy.int32(0)
			
		i += 1		
	a[24] = numpy.int32(data_halo)
	# i'm thinking about not use buffer range any more
	# but we can't change GPU immediately
	#a[25] = numpy.int32(buffer_halo)
	a[25] = numpy.int32(data_halo)
	
	return cuda.In(a)
global FD
FD = []
def run_function(function_package, function_name):
	# global variables
	global FD
	
	# initialize variables
	fp = function_package
	func_output = fp.output
	u, ss, sp = func_output.get_id()
	data_halo = func_output.data_halo
	
	args = fp.function_args
	work_range = fp.work_range

	stream = stream_list[0]
	mod = source_module_dict[function_name]

#	cuda.memcpy_htod_async(bandwidth, numpy.float32(fp.TF_bandwidth), stream=stream)
#	cuda.memcpy_htod_async(front_back, numpy.int32(fp.front_back), stream=stream)

	tf = mod.get_texref('TFF')	
	tf1 = mod.get_texref('TFF1')
	bandwidth,_ = mod.get_global('TF_bandwidth')
	
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
		buffer_range = dp.buffer_range
		buffer_halo = dp.buffer_halo
		
		ad = data_range_to_cuda_in(output_range, full_output_range, buffer_range, buffer_halo=buffer_halo, stream=stream)
		cuda_args += [ad]
		output_package = dp
		
		FD.append(ad)
	else:
		bytes = func_output.data_bytes
		devptr, usage = malloc_with_swap_out(bytes)
		log("created output data bytes %s"%(str(func_output.data_bytes)),'detail',log_type)
		data_range = func_output.data_range
		full_data_range = func_output.full_data_range
		buffer_range = func_output.buffer_range
		buffer_halo = func_output.buffer_halo
	
		ad = data_range_to_cuda_in(data_range, full_data_range, buffer_range, buffer_halo=buffer_halo, stream=stream)

		cuda_args += [ad]
		output_package = func_output
		output_package.set_usage(usage)
		
		if False:
			print "OUTPUT"
			print "OUTPUT_RANGE", data_range
			print "OUTPUT_FULL_RANGE",  full_data_range
			
		FD.append(ad)

	# set work range
	block, grid = range_to_block_grid(work_range)
	cuda_args = [devptr] + cuda_args
	
#	print "GPU", rank, "BEFORE RECV", time.time()
	# Recv data from other process
	for data_package in args:
		u = data_package.get_unique_id()
		data_name = data_package.data_name
		if data_name not in work_range and u != '-1':
			wait_data_arrive(data_package, stream=stream)

#	print "GPU", rank, "Recv Done", time.time()
	# set cuda arguments 
	for data_package in args:
		data_name = data_package.data_name
		data_dtype = data_package.data_dtype
		data_contents_dtype = data_package.data_contents_dtype

		u = data_package.get_unique_id()
		if data_name in work_range:
			cuda_args.append(numpy.int32(work_range[data_name][0]))
			cuda_args.append(numpy.int32(work_range[data_name][1]))
		elif u == '-1':
			data = data_package.data
			dtype = data_package.data_contents_dtype
			if dtype == 'int': 
				cuda_args.append(numpy.int32(data))
			elif dtype == 'float':
				cuda_args.append(numpy.float32(data))
			elif dtype == 'double':
				cuda_args.append(numpy.float64(data))
			else:
				cuda_args.append(numpy.float32(data)) # temp
		else:
			ss = data_package.get_split_shape()
			sp = data_package.get_split_position()
			dp = data_list[u][ss][sp] # it must be fixed to data_package later

			memory_type = dp.memory_type
			if memory_type == 'devptr':

				cuda_args.append(dp.devptr)
				data_range = dp.data_range
				full_data_range = dp.full_data_range
				buffer_range = dp.buffer_range
				
				if False:
					print "DP", dp.info()
					print_devptr(dp.devptr, dp)
				ad = data_range_to_cuda_in(data_range, full_data_range, buffer_range, data_halo=dp.data_halo, buffer_halo=dp.buffer_halo, stream=stream)

				cuda_args.append(ad)
				FD.append(ad)

	# set modelview matrix
	func = mod.get_function(function_name)
	mmtx,_ = mod.get_global('modelview')
	inv_mmtx, _ = mod.get_global('inv_modelview')
	inv_m = numpy.linalg.inv(fp.mmtx)
	cuda.memcpy_htod_async(mmtx, fp.mmtx.reshape(16), stream=stream)
	cuda.memcpy_htod_async(inv_mmtx, inv_m.reshape(16), stream=stream)

	stream_list[0].synchronize()

	if log_type in ['time','all']:
		start = time.time()
	
	kernel_finish = cuda.Event()
	func( *cuda_args, block=block, grid=grid, stream=stream_list[0])
	kernel_finish.record(stream=stream_list[0])
	ctx.synchronize()
#	print "FFFFOOo", func_output.info()
#	print_devptr(cuda_args[0], func_output)
	u, ss, sp = func_output.get_id()
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
		u = data_package.get_unique_id()
		if u != '-1':
			mem_release(data_package)

#	print "Release", time.time()
	return devptr, output_package
	
# set GPU
device_number = rank-int(sys.argv[1])
cuda.init()
dev = cuda.Device(device_number)
ctx = dev.make_context()

# initialization
############################################################
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

s_requests = []

valid_list = []
Event_dict = {}

func_dict = {}
function_code_dict = {} # function
function_and_kernel_mapping = {}
source_module_dict = {}

# data pools
data_pool = []

numpy.set_printoptions(linewidth=200)

Debug = False
GPUDIRECT = True
VIVALDI_BLOCKING = False

log_type = False
n = 2
flag = ''
def init_stream():
	global stream_list
	for elem in range(2):
		stream_list.append(cuda.Stream())
init_stream()

attachment = load_GPU_attachment()
compile_for_GPU(None, kernel_function_name='default')

flag_times = {}
for elem in ["recv","send_order","free","memcpy_p2p_send","memcpy_p2p_recv","request_data","data_package","deploy_code","run_function","wait"]:
	flag_times[elem] = 0
# execution
##########################################################

flag = ''
while flag != "finish":
	if log_type != False: print "GPU:", rank, "waiting"
	source = comm.recv(source=MPI.ANY_SOURCE,    tag=5)
	flag = comm.recv(source=source,              tag=5)
	if log_type != False: print "GPU:", rank, "source:", source, "flag:", flag 
	# interactive mode functions
	if flag == "log":
		log_type = comm.recv(source=source,    tag=5)
		print "GPU:",rank,"log type changed to", log_type
	elif flag == 'say':
		print "GPU hello", rank, name
	elif flag == 'get_function_list':
		if len(function_code_dict) == 0:
			print "GPU:",rank,"Don't have any function"
			continue
		for function_name in function_code_dict:
			print "GPU:", rank, "name:",name, "function_name:",function_name
	elif flag == "remove_function":
		name = comm.recv(source=source,    tag=5)
		if name in function_code_dict:
			print "GPU:",rank, "function:", name, "is removed"
			
			# remove source code
			try:
				del function_code_dict[name]
			except:
				print "Remove Failed1"
				print "function want to remove:", name
				print "exist function_list", function_code_dict.keys()
				continue
			
			# remove kernel function and source modules
			try:
				for kernel_function_name in function_and_kernel_mapping:
					del source_module_dict[kernel_function_name]
			except:
				print "Remove Failed2"
				print "function want to remove:", name
				print "exist function_list", function_and_kernel_mapping.keys()
				continue
			
			# remove mapping between function name and kernel_function_name
			try:
				if name in function_and_kernel_mapping:
					del function_and_kernel_mapping[name]
			except:
				print "Remove Failed3"
				print "function want to remove:", name
				print "exist function_list", function_and_kernel_mapping.keys()
				continue
		else:
			print "GPU:",rank, "No function named:", name
	elif flag == "get_data_list":
		print "GPU:", rank, "data_list:", data_list.keys()
	elif flag == "remove_data":
		uid = comm.recv(source=source,    tag=5)
		if uid in data_list:
			del data_list[uid]
			print "GPU:", rank, "removed data"
	# old functions
	if flag == "synchronize":
		# synchronize
	
		# requests wait
		f_requests = []
		for u in recv_list:
			for ss in recv_list[u]:
				for sp in recv_list[u][ss]:
					f_requests += [elem['request'] for elem in recv_list[u][ss][sp]]

		f_requests += [elem[0] for elem in s_requests]
		MPI.Request.Waitall(f_requests)

		# stream synchronize
		for stream in stream_list: stream.synchronize()

		# context synchronize
		ctx.synchronize()
		comm.send("Done", dest=source, tag=999)
	elif flag == "recv":
		st = time.time()
		data, data_package, request,_ = recv()
		dp = data_package
		notice(dp)
		save_data(data, dp)
		idle()
		flag_times[flag] += time.time() - st
	elif flag == "send_order":
		st = time.time()
		dest = comm.recv(source=source,    tag=53)
		u = comm.recv(source=source,       tag=53)
		ss = comm.recv(source=source,      tag=53)
		sp = comm.recv(source=source,      tag=53)

		log("rank%d, order to send data u=%d to %d"%(rank, u, dest),'general',log_type)

		data, data_package = data_finder(u,ss,sp, gpu_direct)
		dp = data_package
		ctx.synchronize()
		send(data, dp, dest=dest, gpu_direct=gpu_direct)
		idle()
		flag_times[flag] += time.time() - st
	elif flag == "free":
		st = time.time()
		u = comm.recv(source=source,            tag=58)
		ss = comm.recv(source=source,           tag=58)
		sp = comm.recv(source=source,           tag=58)
		data_halo = comm.recv(source=source,    tag=58)

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

			if data_list[u][ss][sp].usage == 0:
				print "Vivaldi system error"
				print "--------------------------------"
				print "Not correct 'usage' value"
				print data_list[u][ss][sp]
				print "--------------------------------"
				assert(False)
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
		dest = comm.recv(source=source,    tag=56)
		task = comm.recv(source=source,    tag=56)

		memcpy_p2p_send(task, dest)
		idle()
		flag_times[flag] += time.time() - st
	elif flag == "memcpy_p2p_recv":
		st = time.time()
		task = comm.recv(source=source,         tag=57)
		data_halo = comm.recv(source=source,    tag=57)
		memcpy_p2p_recv(task, data_halo)

		mem_release(task.source)
		idle()
		flag_times[flag] += time.time() - st
	elif flag == "request_data":
		st = time.time()
		u = comm.recv(source=source,   tag=55)
		ss = comm.recv(source=source,  tag=55)
		sp = comm.recv(source=source,  tag=55)
		log("rank%d, send data u=%d to %d"%(rank, u, source),'general',log_type)
		gpu_direct = comm.recv(source=source, tag=52)
		data, data_package = data_finder(u,ss,sp, gpu_direct)
		send(data, data_package, dest=source, gpu_direct=gpu_direct)
		flag_times[flag] += time.time() - st
	elif flag == "finish":
		disconnect()
	elif flag == 'set_function':
		# x can be bunch of functions or function
		x = comm.recv(source=source, tag=5)
		new_function_code_dict = get_function_dict(x)
		function_code_dict.update(new_function_code_dict)
	elif flag == "run_function":
		function_package = comm.recv(source=source, tag=51)
		try:
			# get kernel function name
			def get_kernel_function_name(function_package):
				kernel_function_name = function_package.function_name
				function_args = function_package.function_args
				for elem in function_args:
					if elem.data_dtype == numpy.ndarray:
						kernel_function_name += elem.data_contents_dtype
				return kernel_function_name		
			kernel_function_name = get_kernel_function_name(function_package)
			
			# compile if kernel function not exist
			if kernel_function_name not in func_dict:
				compile_for_GPU(function_package, kernel_function_name)
				# add mapping between function name and kernel function
				function_name = function_package.function_name
				if function_name not in function_and_kernel_mapping:
					function_and_kernel_mapping[function_name] = []
				if kernel_function_name not in function_and_kernel_mapping[function_name]:
					function_and_kernel_mapping[function_name].append(kernel_function_name)
					
			devptr, output_package = run_function(function_package, kernel_function_name)
			notice(function_package.output)
			save_data(devptr, output_package)
		except:
			exc_type, exc_value, exc_traceback = sys.exc_info()
			a = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
			print a,
		idle()
