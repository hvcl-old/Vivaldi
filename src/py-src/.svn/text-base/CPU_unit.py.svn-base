from mpi4py import MPI
from Vivaldi_cuda_functions import *
import numpy
import time

def request_data_to_CPU_unit(data_name, dst):
	global comm
	rank = comm.Get_rank()
	comm.send(rank,			dest=dst, tag=1)
	comm.send("request",	dest=dst, tag=1)
	comm.send(data_name,    dest=dst,   tag=1)
	recv = comm.recv(source=dst, tag=1)
	if recv == False:
		wait = comm.recv(source=dst, tag=1)
	memory_type = comm.recv(source=dst, tag=1)	
	comm.send("off",		dest=dst, tag=1) #GPUDirect off
	recv = comm.recv(source=dst, tag=1)
	return recv
	pass

def request_data_to_GPU_unit(data_name, dst):
	global comm
	rank = comm.Get_rank()
	comm.send(rank,			dest=dst, tag=1)
	comm.send("request",	dest=dst, tag=1)
	comm.send(data_name,    dest=dst,   tag=1)
	exist = comm.recv(source=dst, tag=1)
	if exist == False:
		wait = comm.recv(source=dst, tag=1)
	memory_type = comm.recv(source=dst, tag=1)
	comm.send("off",		dest=dst, tag=1) #GPUDirect off
	recv = comm.recv(source=dst, tag=1)
	return recv
	pass

def notice(data_name, data_package=[], reset=False):
	global comm
	rank = comm.Get_rank()
	dst = 1
	comm.send(rank,         dest=dst,   tag=1)
	comm.send("notice",     dest=dst,   tag=1)
	comm.send(data_name,	dest=dst,   tag=1)
	comm.send(data_package, dest=dst,   tag=1)
	comm.send(reset,        dest=dst,   tag=1)

def ask_about_var(name):
	global comm
	rank = comm.Get_rank()
	dst = 1
	comm.send(rank,		dest=dst,	tag=1)
	comm.send("ask",	dest=dst,	tag=1)
	comm.send(name,		dest=dst,	tag=1)
	recv = comm.recv(source=dst,	tag=1)
	return recv

def send(data_name, target=None):
	data = data_list[data_name]
	memory_type = 'memory'
	parent.send(memory_type, dest=target, tag=1)
	gpudirect = parent.recv(source=target, tag=1)
	parent.send(data, dest=target, tag=1)

def save_data(data_name, data):
	data_list[data_name] = data
	
	data_package = {}
	data_package['memory_type'] = 'memory'
	data_package['shape'] = None
	data_package['data_range'] = None
	data_package['dtype'] = type(data)
	data_package['byte'] = None
	data_package['device_type'] = 'CPU'

	print "data saved %s and content %s "%(data_name, data_package)

	notice(data_name, data_package)

	if data_name in wait_list:
		target = wait_list[data_name]
		parent.send(True, dest=target, tag=1)
		send(data_name, target)
		del wait_list[data_name]
	pass


























# init mpi
global comm
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = parent.Get_size()
rank = parent.Get_rank()
name = MPI.Get_processor_name()

print "CPU unit rank",rank,"launch at",name

flag = ''
data_list = {}
wait_list = {}

while flag != "finish":
	source = parent.recv(source=MPI.ANY_SOURCE, tag=1)
	print "SOURCE",source
	if type(source) == str:source = int(source)
	flag = parent.recv(source=source, tag=1)
	print "CPU flag",flag,"source:",source,"rank",rank

	if flag == "send":
		data_name = parent.recv(source=source, tag=1)
		print "received data_name:", data_name
		data = parent.recv(source=source, tag=1)
		save_data(data_name, data)

	elif flag == "request":
		data_name = parent.recv(source=source, tag=1) 
		print "requested data_name:", data_name

		exist = data_name in data_list
		parent.send(exist, dest=source, tag=1)
		if data_name in data_list:
			send(data_name, target=source)
		elif data_name not in data_list:
			wait_list[data_name] = source
		pass

	elif flag == "deploy_func":
		print "function deployed"
		code = parent.recv(source=source, tag=1)
		print "summary",code[:20]
		exec code
		pass

	elif flag == "run_func":
		print 'run CPU function'
		func_name = parent.recv(source=source, tag=1)
		print 'name', func_name
		args = parent.recv(source=source, tag=1)
		print 'args', args
		rt	 = parent.recv(source=source, tag=1)
		print 'return name', rt
		work_range = parent.recv(source=source, tag=1)
		print 'work range', work_range
	
		run_range_args = []
		run_range = {}
		size = []
		for elem in work_range:
			t = elem[elem.find("=")+1:]
			left = t[:t.find(":")]
			right = t[t.find(":")+1:]
			run_range[elem[:elem.find("=")]] = (int(left), int(right))
			run_range_args.append( elem[:elem.find("=")])
			size.append( int(right) - int(left))

		# collect need args
		arg_list = []
		for name in args:
			if name in run_range.keys():
				arg_list.append(run_range[name][0])
				arg_list.append(run_range[name][1])
			else:
				# get info about where can i get data
				info = ask_about_var(name)
				if rank not in info:
					for execid in info:
						if info[execid]['device_type'] == 'CPU':
							print 'buf = request_data_to_CPU_unit(\'' + name + '\', ' + str(execid) + ')'
							exec 'buf = request_data_to_CPU_unit(\'' + name  + '\', ' + str(execid) + ')'
							save_data(name, buf)
						elif info[execid]['device_type'] == 'GPU':
							print 'buf = request_data_to_GPU_unit(\'' + name + '\', ' + str(execid) + ')'
							exec 'buf = request_data_to_GPU_unit(\'' + name  + '\', ' + str(execid) + ')'
							save_data(name, buf)
				if name in data_list:
					arg_list.append(data_list[name])

		print rt + ' = ' + func_name + '(*arg_list)'
		print "func args test", arg_list
		for elem in arg_list:
			print type(elem),elem
		sec = time.time()
		exec 'result' + ' = ' + func_name + '(*arg_list)'
		print "cpu function running time"
		sec = time.time() - sec
		print "time.time(): %f " % sec
		print 'result info'
		print len(result), type(result)
		dtype = type(result[0])
		if dtype in (int, float, str, numpy.int64, numpy.int32, numpy.int16, numpy.int8, numpy.float64, numpy.float32): result = numpy.array(result)
		else:
			i = 0
			for elem in result:
				result[i] = result[i].as_array()
				i = i +1
			size.append(len(result[0]))
			result = numpy.array(result)
		print "result buffer size", result.shape
		print "size??", size
		result = result.reshape(size)
		data_list[rt] =  result

		# finish 
		##################################################
		# Let main manager know, functions finish
		print "send function finish", rank
		parent.send("func_finished", dest=source, tag=1)
		# Let memory manager know, data saved here
		# need here?
	

	else:
		pass

parent.Barrier()
parent.Disconnect()
