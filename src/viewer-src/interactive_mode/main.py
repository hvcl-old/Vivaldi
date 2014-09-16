import sys, os, numpy
from mpi4py import MPI

from Vivaldi_load import *
import Vivaldi_install_check
#import Vivaldi_dsl_functions
#from OpenGL.GL import *

# spawn child
def scheduler_update_computing_unit(cud):
	dest = 1
	tag = 5
	comm.isend(rank,                       dest=dest,    tag=tag)
	comm.isend("update_computing_unit",    dest=dest,    tag=tag)
	comm.isend(cud,                        dest=dest,    tag=tag)
def set_device(host, num, type):
	# prepare spawn 	####################################################################
	info = MPI.Info.Create()
	info.Set("host", host)
	
	# spawn	####################################################################
	comm_new = MPI.COMM_SELF.Spawn(
			sys.executable, args=[type+'.py'], maxprocs=num,
			info=info, root=0)

	MPI.COMM_WORLD = comm_new.Merge()
def spawn_all():
	import mpi4py
	# local variables
	MPI4PYPATH = os.path.abspath(os.path.dirname(mpi4py.__path__[0]))
	COMMAND = []
	ARGS = []
	MAXPROCS = []
	INFO = []
	
	global device_info
	device_info = {}
	m_key = ''
	# local_functions
	def read_hostfile():
		# "read hostfile"
		hostfile = Vivaldi_path + '/hostfile/hostfile'
		if '--hostfile' in sys.argv:
			idx = sys.argv.index('--hostfile')
			hostfile = sys.argv[idx+1]
			
		f = open(hostfile)
		x = f.read()
		f.close()		
		return x
	def set_device_info(hostfile):
		my_file = ''

		for line in hostfile.split('\n'):
			line = line.strip()
			if line == '':continue
					
			if '-CPU' in line:
				idx = line.find('=')
				num = line[idx+1:].strip()
				device_info[m_key]['CPU'] = int(num)
			elif '-GPU' in line:
				idx = line.find('=')
				num = line[idx+1:].strip()
				device_info[m_key]['GPU'] = int(num)
			elif '-G' in line:
				idx = line.find('-G')
				GPUDIRECT = line[idx+2:].strip()
			else:
				m_key = line
				device_info[m_key] = {}				
	def add_reader(COMMAND, ARGS, MAXPROCS, INFO):
		# scheduler
		##########################################################################
		filename = "Vivaldi_reader.py"
		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
		info3 = MPI.Info.Create()

		COMMAND += [sys.executable]
		ARGS += [[unit]]
		MAXPROCS += [1]
		INFO += [info3]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_scheduler(COMMAND, ARGS, MAXPROCS, INFO):
		# memory manager
		##########################################################################
		filename = "Vivaldi_memory_manager.py"
		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
		info = MPI.Info.Create()
		
		COMMAND += [sys.executable]
		ARGS += [[unit]]
		MAXPROCS += [1]
		INFO += [info]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_CPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile):
		device_type = 'CPU'
		for host in device_info:
			if device_info[host][device_type] == 0:continue
			filename = 'CPU_unit.py'
			unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
			info = MPI.Info.Create()
			info.Set("host", host)
			
			COMMAND += [sys.executable]
			ARGS += [[unit]]
			MAXPROCS += [device_info[host][device_type]]
			INFO += [info]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_GPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile):
		device_type = 'GPU'
		
		i = 3
		for host in device_info:
			if device_info[host][device_type] == 0:continue
			filename = 'GPU_unit.py'
			unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
			info = MPI.Info.Create()
			info.Set("host", host)
			
			COMMAND += [sys.executable]
			ARGS += [[unit, str(i)]]
			MAXPROCS += [device_info[host][device_type]]
			INFO += [info]
			i = i + device_info[host][device_type]
			
		return COMMAND, ARGS, MAXPROCS, INFO
	def set_computing_unit_list():
		# computing unit start from 3
		# 0: main
		# 1: scheduler
		# 2: reader
		rank = 3
		for host in device_info:
			for device_type in device_info[host]:
				for execid in range(device_info[host][device_type]):
					computing_unit_list[rank] = {'host':host,'device_type':device_type}
					rank = rank + 1	
		
	hostfile = read_hostfile()
	set_device_info(hostfile)
	COMMAND, ARGS, MAXPROCS, INFO = add_scheduler(COMMAND, ARGS, MAXPROCS, INFO)
	COMMAND, ARGS, MAXPROCS, INFO = add_reader(COMMAND, ARGS, MAXPROCS, INFO)
	COMMAND, ARGS, MAXPROCS, INFO = add_CPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile)
	COMMAND, ARGS, MAXPROCS, INFO = add_GPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile)

	new_comm = MPI.COMM_SELF.Spawn_multiple(
			COMMAND, ARGS, MAXPROCS,
			info=INFO, root=0)
	global comm
	comm = new_comm.Merge()
	
	set_computing_unit_list()
	scheduler_update_computing_unit(computing_unit_list)
	
def get_GPU_list(num=-1, type=None):
	# check
	if num == 0:
		print "Vivaldi Warning"
		print "-------------------------------"
		print "Computing unit number should bigger than zero"
		print "-------------------------------"
		assert(False)
	# get list of ranks of GPU process
	execid_list = []
	cnt = 0
	if num == -1: # -1 mean all GPUs
		num = len(computing_unit_list)
	for rank in computing_unit_list:
		if computing_unit_list[rank]['device_type'] == 'GPU':			
			execid_list.append(rank)
			cnt += 1
			if cnt >= num: break
		
	return list(execid_list)
def get_CPU_list(num=-1, type=None):
	print "Not implemented yet"
	assert(False)
	pass
	
# function load
def load_common(filename): # send functions to computing units
	# read file
	def read_file(filename):
		f = open(filename)
		x = f.read()
		f.close()
		return x
	code = read_file(filename)
	# preprocessing to Vivaldi code
	def preprocessing(code):
		from Vivaldi_translator_layer import preprocessing as vp
		code = vp(code)
		return code
	code = preprocessing(code)
	# send code to computing units
	def deploy_function_code(x):
		global comm
		tag = 5
		dest = 1
		comm.isend(0,              dest=dest, tag=tag)
		comm.isend('set_function', dest=dest, tag=tag)
		comm.isend(x,              dest=dest, tag=tag)
	# deploy function 
	deploy_function_code(code)
	# make code to new function dictionary
	def get_function_code_dict(x):
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
	
	new_function_code_dict = get_function_code_dict(code)
	return new_function_code_dict
def load_function(filename): # load function only
	new_function_code_dict = load_common(filename)
	function_code_dict.update(new_function_code_dict)
	
	for elem in new_function_code_dict:
		if elem == 'main':continue
		exec new_function_code_dict[elem] in globals()
def remove_tab_from_main_code(main_code):
	lines = main_code.splitlines()
	new_code = ''
	for line in lines[1:]:
		new_code += line[4:] + '\n'
	
	return new_code
		
def load_main(filename): # load main only
	new_function_code_dict = load_common(filename)
	function_code_dict.update(new_function_code_dict)
	if 'main' in function_code_dict:
		from Vivaldi_translator_layer import parse_main
		main_code = new_function_code_dict['main']
		main_code = parse_main(main_code)
		main_code = remove_tab_from_main_code(main_code)
		exec main_code in globals()
	else:
		print "=============================================="
		print "Vivaldi warning"
		print filename + " don't have main()"
		print "=============================================="
def load_file(filename): # load function and main
	new_function_code_dict = load_common(filename)
	function_code_dict.update(new_function_code_dict)
	
	for elem in new_function_code_dict:
		if elem == 'main':continue
		exec new_function_code_dict[elem] in globals()
		
	if 'main' in function_code_dict:
		from Vivaldi_translator_layer import parse_main
		main_code = new_function_code_dict['main']
		main_code = parse_main(main_code)
		main_code = remove_tab_from_main_code(main_code)
		exec main_code in globals()
		
def remove_function_from_CPU(function_name): # tag
	global comm
	CPU_list = get_CPU_list()
	for dest in CPU_list:
		tag = 5
		comm.isend(0, dest=dest, tag=tag)
		comm.isend('remove_function', dest=dest, tag=tag)
		comm.isend(function_name, dest=dest, tag=tag)
def remove_function_from_GPU(function_name): # tag
	global comm
	GPU_list = get_GPU_list()
	for dest in GPU_list:
		tag = 5
		comm.isend(0, dest=dest, tag=tag)
		comm.isend('remove_function', dest=dest, tag=tag)
		comm.isend(function_name, dest=dest, tag=tag)
def remove_function(function_name): # tag
	remove_function_from_CPU(function_name)
	remove_function_from_GPU(function_name)

# state check function	
def get_function_list(execid_list=[],num=-1): # tag
	global device_info
	global comm
	if execid_list != []:
		for dest in execid_list:
			tag = 5
			comm.send(0, dest=dest, tag=tag)
			comm.send('say', dest=dest, tag=tag)
	elif num != -1:
		pass
	elif execid_list==[] and num == -1: # all
		size = comm.Get_size()
		for i in range(size-1):
			tag = 5
			dest = i + 1
			comm.send(0, dest=dest, tag=tag)
			comm.send('get_function_list', dest=dest, tag=tag)
def get_process_status():
	dest = 1 # scheduler
	tag = 5 # tag
	comm.send(0,                dest=dest,    tag=5)
	comm.send('process_status', dest=dest,    tag=5)

	idle_list = comm.recv(source=dest, tag=5)
	work_list = comm.recv(source=dest, tag=5)
	return idle_list, work_list

# data management
def free_volumes():
	# old version function
	for data_name in data_package_list.keys():
		u = data_list[data_name]
		if u in retain_list:
			for elem in retain_list[u]:
				mem_release(elem)
			del(retain_list[u])

		del(data_package_list[data_name])
def send_data_package(data_package, dest=None, tag=None):
	global comm
	dp = data_package
	t_data, t_devptr = dp.data, dp.devptr
	dp.data, dp.devptr = None, None
	comm.isend(dp, dest=dest, tag=tag)
	dp.data, dp.devptr = t_data, t_devptr
	t_data,t_devptr = None, None
def send_data(dest, data, data_package):
	dp = data_package
	global rank
	global comm
	comm.isend(rank,      dest=dest,    tag=5)
	comm.isend("recv",    dest=dest,    tag=5)
	
	t_data = dp.data
	t_devptr = dp.devptr
	dp.data = None
	dp.devptr = None

	send_data_package(dp, dest=dest,   tag=52)
	dp.data = t_data
	dp.devptr = t_devptr
	t_data = None
	t_devptr = None

	if type(data) == numpy.ndarray:
		request = comm.Isend(data,    dest=dest,    tag=57)
		#global requests
		#requests.append((request, data))
#		MPI.Request.Wait(request)
	else:
		comm.isend(data,              dest=dest,    tag=57)
def scheduler_release(data_package):
	global rank
	global comm
	comm.isend(rank,                dest=1,     tag=5)
	comm.isend("release",           dest=1,     tag=5)
	send_data_package(data_package, dest=1,     tag=57)		
def scheduler_retain(data_package):
	global rank
	global comm
	comm.isend(rank,                dest=1,     tag=5)
	comm.isend("retain",            dest=1,     tag=5)
	send_data_package(data_package, dest=1,     tag=58)
def scheduler_inform(data_package, execid=None):
	dp = data_package
	global rank
	global comm
	dest = 1
	comm.isend(rank,        dest=dest,     tag=5)
	comm.isend("inform",    dest=dest,     tag=5)
	send_data_package(dp,   dest=dest,     tag=55)
	comm.isend(execid,		dest=dest,	   tag=55)
def scheduler_merge(function_package, cnt):
	global comm
	dest = 1
	comm.isend(rank,             dest=dest,    tag=5)
	comm.isend("merge_new",      dest=dest,    tag=5)
	comm.isend(function_package, dest=dest,    tag=5)
	comm.isend(cnt,              dest=dest,    tag=5)
def synchronize():
	comm.isend(rank,             dest=1,    tag=5)
	comm.isend("synchronize",    dest=1,    tag=5)
	comm.recv(source=1,                     tag=999)
	return True
def Vivaldi_gather(data_package):
	dp = data_package
	if not isinstance(dp, Data_package): return dp

	u = dp.unique_id
	if u == None:
		return dp
	
	# ask gathering data 
	comm.isend(rank,			dest=1,		tag=5)
	comm.isend("gather",		dest=1,		tag=5)

	temp1 = dp.data
	temp2 = dp.devptr
	dp.data = None
	dp.devptr = None

	send_data_package(dp,    dest=1,    tag=512)
	dp.data = temp1
	dp.devptr = temp2

	# wati until data created, we don't know where the data will come from
	source = comm.recv(source=MPI.ANY_SOURCE,    tag=5)
	flag = comm.recv(source=source,              tag=5)
	task = comm.recv(source=source,              tag=57)
	halo_size = comm.recv(source=source,         tag=57)
	def recv():
		data_package = comm.recv(source=source,  tag=52)
		dp = data_package
		data_memory_shape = dp.data_memory_shape
		
		dtype = dp.data_contents_memory_dtype
		data = numpy.empty(data_memory_shape, dtype=dtype)
		request = comm.Irecv(data, source=source,tag=57)
		MPI.Request.wait(request)
		
		return data, data_package
	data, data_package = recv()
	return data

def get_file_name(file_name):
	global cnt
	if '.' in file_name:
		file_name, extension = split_file_name_and_extension(file_name)
	else:
		file_name = data_name + '_' +str(cnt)
		cnt += 1
		extension = 'png'
	return file_name, extension.lower()
def save_image_2d(file_name=None, extension='png', buf=None, chan=None):
	if log_type in ['time','all']:
		st = time.time()
	img = None

	if extension == 'raw':
		e = os.system("mkdir -p result/%s"%(file_name))
		f = open('./result/%s.raw'%(file_name), 'wb')
		f.write(buf)
		f.close()
		return

	buf = buf.astype(numpy.uint8)
	if chan == 1:	img = Image.fromarray(buf, 'L')
	elif chan == 3:	img = Image.fromarray(buf, 'RGB')
	elif chan == 4:	img = Image.fromarray(buf, 'RGBA')
		
	e = os.system("mkdir -p result")
	img.save('./result/%s.png'%(file_name), format=extension)
		
	if log_type in ['time','all']:
		t = time.time()-st
		ms = 1000*t
		sp = buf.nbytes/MEGA
		bytes = buf.nbytes
		log("rank%d, \"%s\", save time to hard disk bytes: %.3fMB %.3f ms %.3f MBytes/sec"%(rank, file_name, bytes/MEGA, ms, sp),'time',log_type)
def save_image_3d(file_name=None, extension='dat', buf=None, data_shape=None, chan=None, dtype=None):
	if log_type in ['time','all']:
		st = time.time()
	img = None

	if extension == 'raw':
		e = os.system("mkdir -p result")
		f = open('./result/%s.raw'%(file_name), 'wb')
		f.write(buf)
		f.close()
		return

	if log_type in ['time','all']:
		t = time.time()-st
		ms = 1000*t
		sp = buf.nbytes/MEGA
		bytes = buf.nbytes
		log("rank%d, \"%s\", save time to hard disk bytes: %.3fMB %.3f ms %.3f MBytes/sec"%(rank, file_name, bytes/MEGA, ms, sp),'time',log_type)
def save_image(input1, input2=None, out_of_core=False, normalize=False):
	dtype = 'float32'
	# merge image
	##########################################################################################################

	# get file name 
	if input2 != None: # file name is exist
		data = input1
		file_name = input2
	else: # file name is not exist
		data = input1
		file_name = gen_file_name();

	# data generation
	if isinstance(data, Data_package): 	# data not exist in local
		dp = data
		file_name, extension = get_file_name(file_name)
		dp.file_name = file_name
		dp.extension = extension
		dp.normalize = normalize
		
		if out_of_core:
			dp.out_of_core = True
			reader_save_image_out_of_core(dp)
		else:
			reader_save_image_in_core(dp)	
	elif type(data) == numpy.ndarray: # data exist in local
		shape = data.shape
		n = len(shape)

		file_name, extension = get_file_name(file_name)
		dimension, data_shape, chan = get_dimension_and_channel(shape)

		# if normalize is True, than normalize from 0 ~ 255
		if normalize:
			min = data.min()
			max = data.max()
			if min != max: data = (data - min)*255/(max-min)

		if dimension == 2:
			save_image_2d(file_name=file_name, extension=extension, buf=data, chan=chan)
		elif dimension == 3:
			dtype = python_dtype_to_Vivaldi_dtype(buf.dtype)
			save_image_3d(file_name=file_name, extension=extension, buf=data, data_shape=data_shape, chan=chan, dtype=dtype)
		return

# data management, old functions for compatibility
def VIVALDI_WRITE(data_name, data):
	return data
def VIVALDI_GATHER(data_package):
	if not isinstance(data_package, Data_package): return data_package
	return Vivaldi_gather(data_package)

# reader function
def reader_save_image_out_of_core(data_package):
	dp = data_package
	dest = 2
	comm.isend(rank,                        dest=dest,    tag=5)
	comm.isend("save_image_out_of_core",    dest=dest,    tag=5)
	send_data_package(dp,                   dest=dest,    tag=210)
def reader_save_image_in_core(data_package):
	dp = data_package
	dest = 2
	comm.isend(rank,                        dest=dest,    tag=5)
	comm.isend("save_image_in_core",        dest=dest,    tag=5)
	send_data_package(dp,                   dest=dest,    tag=211)
	
# task functions
from Vivaldi_memory_packages import Data_package, Function_package
def register_function_package(function_package):
	def clear_data():
		# clear volume only
		argument_package_list = function_package.get_args()
		i = 0
		temp_list = []
		for argument_package in argument_package_list:
			temp_list.append(argument_package.data)
			if type(argument_package.data) == numpy.ndarray:
				argument_package.data = None # remove real data
			i += 1
		return temp_list
	temp_data_list = clear_data() # remove real data before mpi send
	global comm
	dest = 1
	comm.isend(0,					dest=dest,    tag=5)
	comm.isend("function",			dest=dest,    tag=5)
	comm.isend(function_package,	dest=dest,    tag=52)
	def recover_data(temp_list):
		i = 0
		argument_package_list = function_package.get_args()
		for argument_package in argument_package_list:
			argument_package.data = temp_list[i]
			i += 1
	recover_data(temp_data_list)
def parallel(function_name='', argument_package_list=[], work_range={}, execid=[],  output_halo=0, output_split={}, merge_func='', merge_order=''):
	# compatibility to old versions
	############################################################
	function_name = function_name.strip()
	def to_range(input):
		dtype = type(input)
		if type(input) == numpy.ndarray:
			input = list(input.shape)
			dtype = type(input)

			n = len(input)

			if input[n-1] in [1,2,3]:
				input.pop()
		
		
		if dtype in [tuple, list]:
			return shape_to_range(input)
		
		if isinstance(input, Data_package):
			dp = input
			work_range = apply_halo(dp.data_range, -dp.data_halo)
			return work_range
		
		if dtype == dict:
			return input
		return {}
	work_range = to_range(work_range)
	
	# input argument error check
	def input_argument_check():
		if type(function_name) != str or function_name == '':
			print "Function_name error"
			print "function_name: ", function_name	
		if function_name not in function_code_dict:
			print "======================================"
			print "Vivaldi Warning"
			print "the function: " + function_name + " not exist"
			print "======================================"
			assert(False)		
		if type(merge_func) != str:
			print "Merge function_name error"
			print "Merge_function name: ", merge_func			
		if type(work_range) != dict:
			print "work_range error"
			print "work_range: ", work_range
			assert(False)
	input_argument_check()
	
	# initialization
	##############################################################
	global unique_id
	# share argument packages
	# and send data to reader
	def share_argument_package_list(arugment_package_list):
		def share_argument_package(argument_package):
			if argument_package.get_unique_id() == '-1': # skip, small variables
				pass
			elif argument_package.shared == False: # not registered variables
				def reader_give_access(data_package):
					scheduler_inform(data_package, 2)
					u = data_package.unique_id

					scheduler_retain(data_package)
					out_of_core = data_package.out_of_core
					if out_of_core: 
						reader_notice_data_out_of_core(data_package)
					else: 
						send_data(2, data_package.data, data_package)
				reader_give_access(argument_package)
		for argument_package in argument_package_list:
			share_argument_package(argument_package)
	share_argument_package_list(argument_package_list)

	# make return package
	def get_return_package(function_name, argument_package_list, work_range, output_halo):
		data_package = Data_package()
		def get_unique_id():
			global unique_id
			unique_id += 1
			return unique_id
		data_package.unique_id = get_unique_id()
		data_package.data_dtype = numpy.ndarray
		data_package.data_halo = output_halo
		
		def get_return_dtype(function_name, argument_package_list):
			from Vivaldi_translator_layer import get_return_dtype
			function_code = function_code_dict[function_name]
			return_dtype = get_return_dtype(function_name, argument_package_list, function_code)
			if return_dtype.endswith('_volume'):
				print "Vivaldi_warning"
				print "---------------------------------"
				print "Check your function"
				print "you are trying to return a volume"
				print "return_dtype: ", return_dtype
				print "---------------------------------"
			return return_dtype
		return_dtype = get_return_dtype(function_name, argument_package_list)
		data_package.set_data_contents_dtype(return_dtype)
		data_package.set_full_data_range(work_range)
		data_package.set_data_range(work_range)
		data_package.halo = output_halo
		data_package.split = output_split
		data_package.shared = True
		return data_package
	return_package = get_return_package(function_name, argument_package_list, work_range, output_halo)
	# register return package to data_package_list
	def register_return_package(key, return_package):
		if key in data_package_list: # cannot happen
			pass
		else:
			data_package_list[key] = return_package
	register_return_package(id(return_package), return_package)
	# register function to scheduler
	def get_function_package(function_name, argument_package_list, return_package, work_range, merge_func='', merge_order=''):
		fp = Function_package()
		fp.set_function_name(function_name)
		fp.set_function_args(argument_package_list)
#		from OpenGL.GL import glGetFloatv, GL_MODELVIEW_MATRIX
#		fp.mmtx					= glGetFloatv(GL_MODELVIEW_MATRIX)
#		fp.inv_mmtx				= numpy.linalg.inv(fp.mmtx)
		fp.work_range = work_range
		#fp.merge_func = merge_func
		#fp.merge_order = merge_order
		fp.output = return_package
		return fp
	function_package = get_function_package(function_name, argument_package_list, return_package, work_range, merge_func, merge_order)
	register_function_package(function_package)
	
	if merge_func != '':
		return_package = return_package.copy()
		return_package.set_data_range( return_package.full_data_range)
		# function name check
		if merge_func not in function_code_dict:
			print "Vivaldi warning"
			print "================================="
			print "function: ",merge_func,"not exist"
			print "================================="
			assert(False)
		# make function package
		merge_function_package = Function_package()
		# set function name
		merge_function_package.set_function_name(merge_func)
		# set return package
		merge_function_package.output = return_package
		# set work_range
		merge_function_package.work_range = return_package.full_data_range
		def get_merge_package_args(return_package, merge_func):
			def get_argument_list(function_name):
				function_code = function_code_dict[merge_func]
				def get_args(name, code):
					idx_start = code.find(name) + len(name) + 1
					idx_end = code.find(')', idx_start)
					args = code[idx_start:idx_end]
					return args.strip()
				function_args = get_args(function_name, function_code)
				if function_args == '':
					print "Vivaldi warning"
					print "================================="
					print "There are no function argument"
					print "================================="
					assert(False)
				argument_list = []
				for arg in function_args.split(','):
					argument_list.append(arg.strip())
				return argument_list
			argument_list = get_argument_list(merge_func)
			argument_package_list = []
			for arg in argument_list:
				argument_package = None
				if arg in AXIS:
					argument_package = Data_package(arg)
				else:
					argument_package = return_package.copy()
				argument_package_list.append(argument_package)
			return argument_package_list
		merge_argument_package_list = get_merge_package_args(return_package, merge_func)
		# set argument
		merge_function_package.set_args(merge_argument_package_list)
		
		# split count
		def get_split_count(argument_package_list):
			cnt = 1
			for argument_package in argument_package_list:
				uid = argument_package.get_unique_id()
				if uid != '-1':
					split = argument_package.split
					for axis in split:
						cnt *= split[axis]
			return cnt
		n = get_split_count(argument_package_list)
		# ask scheduler to merge
		scheduler_merge(merge_function_package, n)
	scheduler_retain(return_package)
	return return_package
def run_function(return_name=None, func_name='', execid=[], work_range=None, args=[], arg_names=[], dtype_dict={}, output_halo=0, halo_dict={}, split_dict={}, merge_func='', merge_order=''): # compatibility to old version
	
	function_name = func_name
	def get_argument_package_list(args, arg_names, split_dict, halo_dict):
		i = 0
		argument_package_list = []
		for data_name in arg_names:
			arg = args[i]
			if data_name in AXIS:
				argument_package = Data_package(data_name)
				argument_package.shared = False
				argument_package_list.append(argument_package)
			else:
				argument_package = None
				split = split_dict[data_name] if data_name in split_dict else {}
				halo = halo_dict[data_name] if data_name in halo_dict else 0 
				
				if isinstance(arg, Data_package):
					argument_package = arg
				else:
					argument_package = Data_package(arg,split=split,halo=halo)
					def get_unique_id(arg):
						if type(arg) != numpy.ndarray:
							return -1
						aid = id(arg)
						if aid in data_package_list:
							return data_package_list[aid].get_unique_id()
						else:
							global unique_id
							unique_id += 1
						return unique_id
					argument_package.unique_id = get_unique_id(arg)
					def add_to_data_package_list(data_package, data):
						if type(data) == numpy.ndarray:
							key = id(data)
							data_package_list[key] = data_package
					add_to_data_package_list(argument_package, arg)
					argument_package.shared = False
				argument_package_list.append(argument_package)
			i += 1
		return argument_package_list
		
	argument_package_list = get_argument_package_list(args, arg_names, split_dict, halo_dict)
	
	def get_output_split(args, arg_names, split_dict, return_name):
		split = {}
		for data_name in split_dict:
			if data_name not in arg_names:
				split = split_dict[data_name]
				break
		
		if return_name in split_dict:
			split = split_dict[return_name]
		return split
	output_split = get_output_split(args, arg_names, split_dict, return_name)
	
	return parallel(function_name, argument_package_list, work_range, execid, output_halo, output_split,  merge_func, merge_order)
	
# test function
def say(execid_list=[],num=-1):
	global device_info
	global comm
	if execid_list != []:
		for dest in execid_list:
			tag = 5
			comm.send(0, dest=dest, tag=tag)
			comm.send('say', dest=dest, tag=tag)
	elif num != -1:
		pass
	elif execid_list==[] and num == -1: # all
		size = comm.Get_size()
		for i in range(size-1):
			tag = 5
			dest = i + 1
			comm.send(0, dest=dest, tag=tag)
			comm.send('say', dest=dest, tag=tag)

# initialize variables
try:
	cache
	print "Vivaldi_init not initialize variables"
except:
	cache = {}
	print "Vivaldi_init initialize variables"
	Vivaldi_path = os.environ.get('vivaldi_path')
	DATA_PATH = Vivaldi_path + '/data'
	comm = MPI.COMM_WORLD
	rank = 0
	cnt = 0
	
	# spawn child
	device_info = {} # host, device_type
	computing_unit_list = {} # mapping rank and process
				
	# task functions
	# data_package_list have to type of id. 
	# one is id(data), when real data exist
	# another is id(data_package), when real data not exist
	data_package_list = {} 
	function_code_dict = {}
	
	AXIS = ['x','y','z','w']
	unique_id = -1
	
	x = 'x'
	y = 'y'
	z = 'z'
	w = 'w'
	spawn_all()
	
# argument parsing
def Vivaldi_input_argument_parsing(argument_list):
	def option_parsing():
		if '-L' in argument_list:
			idx = argument_list.index('-L')
			GPUDIRECT = argument_list[idx+1]
	option_parsing()
	def get_filename(argument_list):
		filename = None
		skip = False
		i = 0
		for elem in argument_list[1:]:
			i += 1
			if skip:
				skip = False
				continue
			if elem.startswith('-'):
				skip = True
			else:
				filename = argument_list[i]		
		return filename
	filename = get_filename(argument_list)
	if filename != None:
		load_file(filename)
import sys
Vivaldi_input_argument_parsing(sys.argv)
#say()
# interactive mode functions
def interactive_mode():
	# interactive mode 
	################################################################################################
	import os
	import sys
	from code import InteractiveConsole
	from tempfile import mkstemp

	EDITOR = os.environ.get('EDITOR', 'vi')
	EDIT_CMD = '\e'

	class VivaldiInteractiveConsole(InteractiveConsole):
		def __init__(self, *args, **kwargs):
			self.last_buffer = [] # This holds the last executed statement
			InteractiveConsole.__init__(self, *args, **kwargs)

		def runsource(self, source, *args):
			from Vivaldi_translator_layer import line_translator
			source = line_translator(source, data_package_list)
	#		print "SSS", source
			self.last_buffer = [ source.encode('latin-1') ]
			return InteractiveConsole.runsource(self, source, *args)

		def raw_input(self, *args):
			line = InteractiveConsole.raw_input(self, *args)
		#	print "DDDD", line
			return line
	def Vivaldi_interactive(_globals, _locals):
		"""
		Opens interactive console with current execution state.
		Call it with: `console.open(globals(), locals())`
		"""
		import readline
		import rlcompleter

		context = _globals
		context.update(_locals)
		readline.set_completer(rlcompleter.Completer(context).complete)
		shell = VivaldiInteractiveConsole(context)
		shell.interact()

	Vivaldi_interactive(globals(), locals())

if 'main' not in function_code_dict:
	interactive_mode()