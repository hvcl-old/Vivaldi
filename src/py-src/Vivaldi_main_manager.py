# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
sys.path.append(VIVALDI_PATH+'/src')
from common import *
#####################################################################

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

from Vivaldi_cuda_functions import *
from Vivaldi_load import *
from Vivaldi_misc import *
from Vivaldi_functions import Vivaldi_functions
from Vivaldi_memory_packages import Data_package, Function_package

DATA_PATH = VIVALDI_PATH + '/data/'

try:
	from PyQt4 import QtGui, QtCore, QtOpenGL, Qt
	from PyQt4.QtOpenGL import QGLWidget
	from OpenGL.GL import *

	# Edit by Anukura $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	import sys, time
	sys.path.append(VIVALDI_PATH + "/src/viewer-src")

	import Vivaldi_viewer
	from Vivaldi_viewer import enable_viewer
except:
	pass

viewer_on = False
trans_on = False

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# match execid and compute unit type
computing_unit_list = {}
data_list = {None:-1}
retain_list = {}
data_package_list = {}

requests = []

unique_id = 0
mmtx = numpy.eye(4,dtype=numpy.float32)
inv_mmtx = numpy.eye(4,dtype=numpy.float32)


#	 Matrix functions
##################################################################################################

def LoadMatrix(name):
	if name == "MODELVIEW":
		pass
	pass

def LoadIdentity():
	global mmtx
	global inv_mmtx
	mmtx = numpy.eye(4,dtype=numpy.float32)
	inv_mmtx = numpy.eye(4,dtype=numpy.float32)

def Rotate(angle, x, y, z):
	import math
	pi = math.pi

	l = x*x + y*y + z*z
	l = 1/math.sqrt(l)
	x = x*l
	y = y*l
	z = z*l

	#matrix
	th = math.pi/180*(angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse
	th = math.pi/180*(-angle)
	c = math.cos(th)
	s = math.sin(th)
	tm = numpy.array([ x*x*(1-c)+c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0, x*y*(1-c)+z*s, y*y*(1-c)+c, y*z*(1-c)-x*s, 0, x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z*(1-c)+c,0, 0,0,0,1], dtype=numpy.float32)
	tm = tm.reshape((4,4))
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)
	log("ROTATE\n"+str(mmtx)+'\n'+str(inv_mmtx)+'\n'+str(numpy.dot(inv_mmtx,mmtx)),"detail", log_type)

def Translate(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = x
	tm[1][3] = y
	tm[2][3] = z
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][3] = -x
	tm[1][3] = -y
	tm[2][3] = -z
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)
	log("TRANSLATE\n"+str(mmtx)+'\n'+str(inv_mmtx)+'\n'+str(numpy.dot(inv_mmtx,mmtx)),'detail', log_type)

def Scaled(x, y, z):
	#matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = x
	tm[1][1] = y
	tm[2][2] = z
	global mmtx
	mmtx = numpy.dot(mmtx, tm)


	#inverse matrix
	tm = numpy.eye(4,dtype=numpy.float32)
	tm[0][0] = 1.0/x
	tm[1][1] = 1.0/y
	tm[2][2] = 1.0/z
	global inv_mmtx
	inv_mmtx = numpy.dot(tm, inv_mmtx)
	log("SCALED\n"+str(mmtx)+'\n'+str(inv_mmtx)+'\n'+str(numpy.dot(inv_mmtx,mmtx)),'detail', log_type)

def free_volumes():
	for data_name in data_package_list.keys():
		u = data_list[data_name]
		if u in retain_list:
			for elem in retain_list[u]:
				mem_release(elem)
			del(retain_list[u])

		del(data_package_list[data_name])

def VIVALDI_WRITE(data_name, data):

	if data_name in data_package_list:
		u = data_list[data_name]
		if u in retain_list:
			for elem in retain_list[u]:
				mem_release(elem)
			del(retain_list[u])

		del(data_package_list[data_name])

	if isinstance(data, Data_package):
		dp = data
		dp.data_name = data_name

		u = dp.unique_id
		if u == None:
			globals()[data_name] = data
			manage_as_data_package(data_name)
			reader_give_access_to_data(data_name)
			dp = data_package_list[data_name]

			u = dp.unique_id
			if u not in retain_list: retain_list[u] = []
			retain_list[u].append(dp.copy())

		u = dp.unique_id

		data_list[data_name] = u
		data_package_list[data_name] = dp

		data = dp
	else:
		pass

	return data

def manage_as_data_package(data_name):
	# bring data from global variable dictionary
	data = globals()[data_name]

	# make data package for manage
	temp = Data_package()
	
	if type(data) == numpy.ndarray:
		# case of data is numpy.ndarray
		dtype = data.dtype
		shape = list(data.shape)
		n = len(shape)
		last = shape[n-1]
		# for make Vivaldi data type name
		type_name,shape = make_type_name(dtype, shape, last)
		temp.data_dtype = numpy.ndarray
		temp.set_data_contents_dtype(type_name)
		data_range = shape_to_range(shape)
		
		temp.set_data_range(data_range)
		temp.set_full_data_range(data_range)

		temp.data = copy.copy(data)
		
	elif isinstance(data, Data_package):
		temp = data.copy()
		
	temp.data_name = data_name
	global unique_id
	u = unique_id
	unique_id += 1

	temp.unique_id = u

	data_list[data_name] = u
	data_package_list[data_name] = temp

# misc function
##################################################################################################
def send_data_package(data_package, dest=None, tag=None):
	dp = data_package
	t_data, t_devptr = dp.data, dp.devptr
	dp.data, dp.devptr = None, None
	comm.isend(dp, dest=dest, tag=tag)
	dp.data, dp.devptr = t_data, t_devptr
	t_data,t_devptr = None, None
	
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

def to_split(input):

	dtype = type(input)
	if dtype == numpy.ndarray:
		input = list(numpy.array(a).reshape(-1,))
		dtype = type(input)

	if dtype in [tuple, list]:
		temp = shape_to_range(input)
		split = {}
		for axis in AXIS:
			if axis in temp: split[axis] = temp[axis][1]
			else: split[axis] = 1
		return split
		
	if dtype == dict:
		for axis in AXIS:
			if axis not in input:input[axis] = 1
		return input

	return {}

# main manager function
#################################################################################################
global temp_data_range_list
temp_data_range_list = None

def run_function(return_name=None, func_name='', execid=[], work_range=None, args=[], dtype_dict={}, output_halo=0, halo_dict={}, split_dict={}, merge_func='', merge_order=''):

	def input_argument_check():
	
		if type(func_name) != str:
			print "Function_name error"
			print "Func_name: ", func_name
	
		if type(dtype_dict) != dict:
			print "Dtype_dict error"
			print "Dtype_dict: ", dtype_dict
			assert(False)
		
		if type(split_dict) != dict:
			print "Split_dict error"
			print "Split_dict: ", split_dict
			assert(False)
		
		if type(halo_dict) != dict:
			print "Halo_dict error"
			print "Halo_dict: ", halo_dict
			assert(False)
			
		if type(merge_func) != str:
			print "Merge function_name error"
			print "Merge_function name: ", merge_func
		
	# compatibility
	############################################################
	function_name = func_name
	args_list = args
	
	modifier_dict={}

	# input argument error check
	input_argument_check()
	
	# initialization
	##############################################################
	global mmtx, inv_mmtx
	global unique_id
	function_package = Function_package()
	fp = function_package

	if Vivaldi_viewer.v != None:
		mmtx = Vivaldi_viewer.mmtx
		inv_mmtx = Vivaldi_viewer.inv_mmtx
	
	Debug = False
	
	# arguments
	##################################################################################
	new_args = []
	for data_name in args_list:	
		if data_name in data_package_list:
			# data_name, is managed as data package in the main manager
			dp = data_package_list[data_name]
			dtype = str(dp.data_contents_dtype)
			dtype = dtype.replace('_volume','')
			function_name += dtype
			# we should give access to data to reader, before running function
			# check it is already available from reader or not

			data = globals()[data_name]

			u = dp.unique_id

			flag = False
			if dp.out_of_core and u not in retain_list: flag = True # out of core and Not informed to memory manager
			if not dp.out_of_core and u not in retain_list: flag = True # in core and Didn't informed ( function output already informed)
			if flag:
				manage_as_data_package(data_name)
				dp = data_package_list[data_name]

				u = dp.unique_id
				if u not in retain_list: retain_list[u] = []
				retain_list[u].append(dp.copy())
	
				reader_give_access_to_data(data_name)

			dp = dp.copy()
			dp.data = None
			dp.devptr = None

		elif data_name in globals():
			# There are two kinds of data here
			# 1. volume
			# 2. values
			
			data = globals()[data_name]
			
			if type(data) == numpy.ndarray: # this is volume			
				# now the data is also managed as data package 
				manage_as_data_package(data_name)
				# now we have data_package correspond to the data
				dp = data_package_list[data_name]

				u = dp.unique_id
				if u not in retain_list: retain_list[u] = []
				retain_list[u].append(dp.copy())
		
				# Vivaldi reader have access to this data
				reader_give_access_to_data(data_name)
		
				# than make a new function name using the existing function name, the data_name and data dtype
			
				dtype = str(dp.data_contents_dtype)
				dtype = dtype.replace('_volume','')
				
				function_name += str(dp.data_contents_dtype)
				dp = dp.copy()
				dp.data = None
				dp.devptr = None
			else: # this is constant
				dp = Data_package()

				dtype = type(data_name)
	
				dp.data_name = data_name
				dp.unique_id = -1
				dp.data_dtype = dtype
				dp.data_contents_dtype = dtype
				dp.data_contents_memory_dtype = dtype
				dp.data = data
		else:
			# data_name not in the globals list
			# it is usually AXIS or constant like x,y not previously defined
			if isinstance(data_name, Data_package):
				dp = data_name
			else:
				data = None
				dp = Data_package()

				dtype = type(data_name)
	
				dp.data_name = data_name
				dp.unique_id = -1
				dp.data_dtype = dtype
				dp.data_contents_dtype = dtype
				dp.data_contents_memory_dtype = dtype
				dp.data = data_name
		
		new_args.append(dp)
	args_list = new_args	

	# get Vivaldi functions
	######################################################################################
	global parsed_Vivaldi_functions
	func_args = args_list

	return_dtype = parsed_Vivaldi_functions.get_return_dtype(function_name)
	fp.set_function_name(function_name)
	fp.output.unique_id					= unique_id
	fp.mmtx								= mmtx
	fp.inv_mmtx							= inv_mmtx
	fp.output.data_dtype				= numpy.ndarray
	fp.output.data_name					= return_name

	if return_dtype == '':
		print "======================================================="
		print "VIVALDI ERROR, can not find return dtype"
		print "function_name:", function_name
		print "return name:", return_name
		print "return dtype:", return_dtype
		print "======================================================="
		assert(False)
	fp.output.set_data_contents_dtype(return_dtype)

	v = Vivaldi_viewer.v
	trans_on = Vivaldi_viewer.trans_on
	transN = Vivaldi_viewer.transN

	if trans_on == True:
		if v.getIsTFupdated() == 1:
			fp.trans_tex 		  = v.getTFF()
			fp.update_tf = 1
			fp.update_tf2 = 0
			v.TFF.widget.updated = 0
		elif v.getIsTFupdated2() == 1:
			fp.trans_tex 		  = v.getTFF2()
			fp.update_tf = 0
			fp.update_tf2 = 1
			v.TFF2.widget.updated = 0
	
		fp.TF_bandwidth		  = v.getTFBW()
		fp.CRG = v.window.CRG

	output_halo = 0
	if type(work_range) == dict:
		if 'work_range' in work_range:
			work_range = work_range['work_range']
	
	if return_name != None:
		# merge_func
		###############################################################
		func_args = ['front','back']
		func_dtypes = {}
		for elem in func_args:
			func_dtypes[ elem ] = return_dtype
		
		new_name = make_func_name_with_dtypes(merge_func, func_args, func_dtypes)
		merge_func = new_name
		# execid
		###################################################################################
		if isinstance(execid, Data_package): execid = execid.data
		if type(execid) != list: execid = [execid]
		execid_list = execid
		fp.execid_list                    = execid

		# work range
		##################################################################################
		if type(work_range) == dict and work_range == {}:
			for data_name in args_list:
				if isinstance(data_name, Data_package):
					if dp.unique_id == -1:continue
					data_name = dp.data_name
					dp = data_package_list[data_name]
					work_range = dp.full_data_range
					break
		

		work_range = to_range(work_range) 
	
	if return_name == '':
		return_name = None
		work_range = {'work_range':work_range}

		args = [return_name, function_name, execid, work_range, args_list, dtype_dict, output_halo, halo_dict, split_dict, merge_func, merge_order]

		return None, run_function, args
		#return None

	# local functions
	############################################################################
	def make_tasks2(arg_packages, i):
		global unique_id
		if i == len(args_list):
		
			# common variables
			global unique_id
			fp.function_args					= arg_packages
			
			modifier = modifier_dict['output']
			
			if decom == 'in_and_out_split1':
				num = modifier['num']
				work_range = modifier['range_list'][num-1]
				fp.work_range = work_range
				
				split = modifier['split']
				data_halo = modifier['data_halo']
				buffer_halo = modifier['buffer_halo']
				full_data_range = modifier['data_range']
				
				fp.output.data_halo = data_halo
				split_position = make_split_position(split, num)
				fp.output.split_position = str(split_position)
				data_range = apply_halo(work_range, data_halo)
		
				fp.output.set_data_range(str(data_range))
				fp.output.set_full_data_range(str(full_data_range))
				fp.output.set_buffer_range(buffer_halo)
			
				modifier['num'] += 1
			elif decom == 'in_and_out_split2':
				num = modifier['num']
				work_range = modifier['range_list'][num-1]
				fp.work_range = work_range
				
				split = modifier['split']
				data_halo = modifier['data_halo']
				buffer_halo = modifier['buffer_halo']
				full_data_range = modifier['data_range']
				
				fp.output.data_halo = data_halo
				split_position = make_split_position(split, num)
				fp.output.split_position = str(split_position)
				data_range = apply_halo(work_range, data_halo)
		
				fp.output.set_data_range(str(data_range))
				fp.output.set_full_data_range(str(full_data_range))
				fp.output.set_buffer_range(buffer_halo)
			
				modifier['num'] += 1
			elif decom == 'in':
				fp.output.unique_id = unique_id
				
				output_range = apply_halo(output_range_list[0], output_halo)
				fp.output.set_data_range(output_range)
				fp.output.split_shape = str(SPLIT_BASE)
				fp.output.split_position = str(SPLIT_BASE)
				fp.work_range = output_range
				
				# buffer
				modifier = modifier_dict['output']
				buffer_halo = modifier['buffer_halo']
				fp.output.set_buffer_range(buffer_halo)
			elif decom == 'out':
				num = modifier['num']
				work_range = modifier['range_list'][num-1]
				fp.work_range = work_range
				
				split = modifier['split']
				data_halo = modifier['data_halo']
				buffer_halo = modifier['buffer_halo']
				full_data_range = modifier['data_range']
				
				fp.output.data_halo = data_halo
				split_position = make_split_position(split, num)
				fp.output.split_position = str(split_position)
				data_range = apply_halo(work_range, data_halo)
		
				fp.output.set_data_range(str(data_range))
				fp.output.set_full_data_range(str(full_data_range))
				fp.output.set_buffer_range(buffer_halo)
			
				modifier['num'] += 1

			u = fp.output.unique_id
			unique_id += 1
			mem_retain(fp.output)
			if u not in retain_list: retain_list[u] = []
			retain_list[u].append(fp.output.copy())
			register_function(execid, fp)
		
			return

		dp = args_list[i]
		data_name = dp.data_name
		dp.memory_type = 'memory'

		# normal variables
		if dp.unique_id != -1:
			# setting about full data
			dp.split_shape = str(SPLIT_BASE)
			buf = dp.data
			# replace copy of original
			dp = dp.copy()

			if decom == 'in_and_out_split1':
				"""
					input output decomposition
				"""
				global in_and_out_n
				
				u = dp.unique_id
				data_name = dp.data_name
				modifier = modifier_dict[data_name] if data_name in modifier_dict else {}


				# split shape
				split_shape = modifier['split']
				dp.split_shape = str(split_shape)
				range_list = data_range_list_dict[data_name]
				data_halo = modifier['data_halo']
				buffer_halo = modifier['buffer_halo']
				
				cnt = modifier['cnt']
		
				split_position = make_split_position(split_shape, in_and_out_n)
				dp.split_position = str(split_position)
		
				# set data_range
				data_range = apply_halo(range_list[in_and_out_n-1], data_halo, dp.full_data_range)
				dp.set_data_range(data_range)
				dp.data_halo = data_halo
				
				# set buffer halo
				buffer_range = apply_halo(range_list[in_and_out_n-1], buffer_halo)
				dp.set_buffer_range(buffer_range)
				dp.buffer_halo = buffer_halo
				
				if Debug:
					print "In and out DP", dp
				make_tasks2(arg_packages + [dp], i + 1)
			elif decom == 'in_and_out_split2':
				u = dp.unique_id
				data_name = dp.data_name
				modifier = modifier_dict[data_name] if data_name in modifier_dict else {}
				# split shape
				
				split_shape = modifier['split']
				dp.split_shape = str(split_shape)
				# make splited data and go to next argument

				data_name = dp.data_name
				data_halo = modifier['data_halo']
				data_range_list = data_range_list_dict[data_name]

				buffer_halo = modifier['buffer_halo']
				n = 1
				dp.data_halo = data_halo
				for data_range in data_range_list:
					data_range = apply_halo(data_range, data_halo)
					
					dp.data_dtype = numpy.ndarray
					dp.set_data_range(data_range)
					dp.data_halo = data_halo
					
					memory_shape = dp.data_memory_shape
					shape = dp.data_shape
					bytes = dp.data_bytes
	
					# make depth
					depth = make_depth(data_range, mmtx)
					dp.depth = depth
					fp.output.depth = depth
					mem_depth(u, str(SPLIT_BASE), str(SPLIT_BASE), depth)

					split_position = make_split_position(split_shape, n)
					n += 1
					dp.split_position = str(split_position)
					mem_depth(data_list[data_name], str(split_shape), dp.split_position, depth)

					dp.set_buffer_range(buffer_halo)
					make_tasks2(arg_packages + [dp], i + 1)
			elif decom == 'in':
				"""
					input decomposition
					data range is same
				"""
				u = dp.unique_id
				data_name = dp.data_name
				modifier = modifier_dict[data_name] if data_name in modifier_dict else {}
				# split shape
				
				split_shape = modifier['split']
				dp.split_shape = str(split_shape)
				# make splited data and go to next argument

				data_name = dp.data_name
				data_halo = modifier['data_halo']
				data_range_list = data_range_list_dict[data_name]

				buffer_halo = modifier['buffer_halo']
				n = 1
				dp.data_halo = data_halo
				for data_range in data_range_list:
					data_range = apply_halo(data_range, data_halo)
					
					dp.data_dtype = numpy.ndarray
					dp.set_data_range(data_range)
					dp.data_halo = data_halo
					
					memory_shape = dp.data_memory_shape
					shape = dp.data_shape
					bytes = dp.data_bytes
	
					# make depth
					depth = make_depth(data_range, mmtx)
					dp.depth = depth
					fp.output.depth = depth
					mem_depth(u, str(SPLIT_BASE), str(SPLIT_BASE), depth)

					split_position = make_split_position(split_shape, n)
					n += 1
					dp.split_position = str(split_position)
					mem_depth(data_list[data_name], str(split_shape), dp.split_position, depth)

					dp.set_buffer_range(buffer_halo)
					if Debug:
						print "DP", dp
					make_tasks2(arg_packages + [dp], i + 1)
			elif decom == 'out':
				u = dp.unique_id
				# basic package setting
				dp.split_shape = str(SPLIT_BASE)
				dp.split_position = str(SPLIT_BASE)
				
				data_name = dp.data_name
				modifier = modifier_dict[data_name] if data_name in modifier_dict else {}

				range_list = data_range_list_dict[data_name]

				data_halo = modifier['data_halo']
				buffer_halo = modifier['buffer_halo']
				
				# data_range
				data_range = apply_halo(range_list[0], data_halo)
				dp.set_data_range(data_range)
				dp.data_halo = data_halo
				
				dp.set_full_data_range(data_range)
				
				# buffer range
				buffer_range = apply_halo(range_list[0], buffer_halo)
				dp.set_buffer_range(buffer_range)
				dp.buffer_halo = buffer_halo
						
				make_tasks2(arg_packages + [dp], i + 1)
		else:
			make_tasks2(arg_packages + [dp], i + 1)
	def check_in_and_out(modifier_dict, input_cnt, output_cnt):
		flag = False
		in_and_out_split = True
		
		# in_and_out spilt version1 test
		
		# all split shape is identical
		for data_name in modifier_dict:
			modifier = modifier_dict[data_name]
			if flag:
				if split == modifier['split']:
					pass
				else:
					in_and_out_split = False
					break
			else:
				# first one skip
				split = modifier['split']
				flag = True
		
		if in_and_out_split:
			return 'in_and_out_split1'
		
		# in_and_out split version2 test
		
		# same number of input and output count
		if output_cnt == input_cnt:
			return 'in_and_out_split2'
			
		return False
	############################################################################

	# make argument name list
	args_name_list = []
	for elem in args_list:
		if isinstance(elem, Data_package):
			args_name_list.append(elem.data_name)

	if return_name == None:
		return_name = 'output'

	# set output information
	modifier_dict['output'] = {}
	output_split = split_dict[return_name] if return_name in split_dict else SPLIT_BASE

	output_data_range = work_range
	output_data_halo = 0

	buffer_halo = 10

	output_dtype = return_dtype
	output_range = to_range(output_data_range)
	output_split = to_split(output_split)
	output_range_list = make_range_list(output_range, output_split)

	cnt = shape_to_count(output_split)

	modifier_dict['output']['split'] = output_split
	modifier_dict['output']['data_range'] = output_range
	modifier_dict['output']['data_halo'] = output_halo
	modifier_dict['output']['cnt'] = cnt
	modifier_dict['output']['buffer_halo'] = buffer_halo
	modifier_dict['output']['num'] = 1
	modifier_dict['output']['range_list'] = output_range_list
	output_cnt = cnt

	
	# temp data package
	temp = Data_package()
	temp.data_name = return_name
	temp.unique_id = unique_id
	temp.data_dtype = numpy.ndarray
	temp.data_halo = output_halo
	temp.set_data_contents_dtype(return_dtype)

	# modifier information about input
	input_cnt = 1

	data_range_list_dict = {}
	
	# make modifiers_list for each argument
	for args in args_list:
		name = args.data_name
		if args.unique_id != -1:
			modifier = {}

			modifier['data_range'] = args.data_range
			modifier['dtype'] = args.data_contents_dtype

			data_range = args.data_range
			data_halo = args.data_halo
			data_range = apply_halo(data_range, -data_halo)

			split = split_dict[name] if name in split_dict else SPLIT_BASE

			for axis in AXIS:
				if axis not in split:
					split[axis] = 1
			
			data_range_list = make_range_list(data_range, split)

			data_range_list_dict[name] = data_range_list
			cnt = shape_to_count(split)

			modifier_dict[name] = {}
			modifier_dict[name]['split'] = split
			modifier_dict[name]['data_range'] = data_range
			modifier_dict[name]['data_halo'] = halo_dict[name] if name in halo_dict else 0
			modifier_dict[name]['buffer_halo'] = buffer_halo
			
			modifier_dict[name]['cnt'] = cnt	
			input_cnt *= cnt

	in_and_out_split = check_in_and_out(modifier_dict, input_cnt, output_cnt)
	
	if in_and_out_split == 'in_and_out_split1':
		decom = 'in_and_out_split1'
		# this is special case called in&out split
		fp.output.split_shape = str(output_split)
		
		global in_and_out_n
		in_and_out_n = 1
		for work_range in output_range_list:
			make_tasks2([], 0)
			in_and_out_n += 1

		modifier = modifier_dict['output']
		data_halo = modifier['data_halo']	
		full_data_range = apply_halo(output_range, data_halo)
		
		temp.set_data_range(str(full_data_range))
		temp.set_full_data_range( str(full_data_range))
		temp.set_buffer_range(str(full_data_range))
		temp.data_halo = data_halo

		unique_id += 1
	#	print "TEMP", temp
		return temp
	elif in_and_out_split == 'in_and_out_split2':
		decom = 'in_and_out_split2'
		
		fp.output.split_shape = str(output_split)
		
		make_tasks2([], 0)
		
		modifier = modifier_dict['output']
		data_halo = modifier['data_halo']
		full_data_range = apply_halo(output_range, data_halo)
		
		temp.set_data_range(str(full_data_range))
		temp.set_full_data_range( str(full_data_range))
		temp.set_buffer_range(str(full_data_range))
		temp.data_halo = data_halo
		
		unique_id += 1
		return temp
	elif input_cnt > 1:
		"""
			input decomposition
		"""
		decom = 'in'
		count = input_cnt
		# set function package output
		full_data_range = apply_halo(output_range, output_halo)

		fp.output.set_data_range( dict(full_data_range))
		fp.output.set_full_data_range( dict(full_data_range))
		fp.output.data_halo = output_halo

		# set output package
		temp.set_data_range( str(full_data_range))
		temp.set_full_data_range( str(full_data_range))
		temp.set_buffer_range(str(full_data_range))
		# register intermediate merge function
		u = unique_id
		inter = range(unique_id+1, unique_id+count-1)
		
		for inter_id in inter:
			temp.unique_id = inter_id
			mem_retain(temp)
		
		unique_id += count-1
		temp.unique_id = u

		# make input functions 
		make_tasks2([], 0)

		out_range = range(unique_id, unique_id+count)
		
		# intermediate merge functions 
		dimension = len(output_range)
		scheduler_request_merge(temp, out_range, merge_func, merge_order, dimension)

		#mem_inform(temp)
		return temp
	elif output_cnt > 1:
		"""
			output decomposition
		"""
		decom = 'out'
		fp.output.split_shape = str(output_split)
		fp.output.data_halo = output_halo
		
		full_data_range = apply_halo(output_range, output_halo)
		
		n = 1
		for work_range in output_range_list:
			split_position = make_split_position(output_split, n)
			n += 1
			fp.output.split_position = str(split_position)
			data_range = apply_halo( work_range, output_halo)
			fp.output.set_data_range( str(data_range))
			fp.output.set_full_data_range( str(full_data_range))
			fp.output.set_buffer_range(buffer_halo)
			
			make_tasks2([], 0)
	
		temp.set_data_range(str(full_data_range))
		temp.set_full_data_range( str(full_data_range))
		temp.set_buffer_range(str(full_data_range))
		
		unique_id += 1
		return temp
	else:
		# split input and output both, but not in&out split

		print "==============================="
		print "VIVALDI ERROR"
		print "tried to split input and output together but number of input split and output split is different"
		print "input_cnt: ", input_cnt
		print "output_cnt: ", output_cnt
		print "==============================="
		assert(False)

	assert(False)

# process list
def get_processor_list():
	temp = {}
	pname = MPI.Get_processor_name()
	temp[0] = {'machine':pname, 'type':'main'}
	temp[1] = {'machine':pname, 'type':'main_manager'}
	temp[2] = {'machine':pname, 'type':'memory_manager'}
	temp[3] = {'machine':pname, 'type':'reader'}
	cul = computing_unit_list
	for rank in cul:
		temp[rank] = {'machine':cul[rank]['computer'], 'type':cul[rank]['device_type']}

	return temp

# scheduler related functions
def scheduler_request_merge(data_package, rg, func, order, dimension):
	comm.isend(rank,		dest=2,		tag=2)
	comm.isend("merge",		dest=2,		tag=2)
	send_data_package(data_package, dest=2, tag=23)
	comm.isend(rg,			dest=2,		tag=23)
	comm.isend(func,		dest=2,		tag=23)
	comm.isend(order,		dest=2,		tag=23)
	comm.isend(dimension,	dest=2,		tag=23)
	
	# Anukura
	global v
#	comm.isend(v.getFB(),	dest=2,		tag=23)
	comm.isend(0,		dest=2,		tag=23)

# reader related function
def reader_save_image_out_of_core(data_package):
	dp = data_package
	dest = 2
	comm.isend(rank,						dest=dest,	tag=2)
	comm.isend("save_image_out_of_core",	dest=dest,	tag=2)
	send_data_package(dp,	dest=dest,	tag=210)

def reader_save_image_in_core(data_package):
	dp = data_package
	dest = 2
	comm.isend(rank,						dest=dest,	tag=2)
	comm.isend("save_image_in_core",		dest=dest,	tag=2)
	send_data_package(dp,	dest=dest,	tag=211)

def reader_notice_data_out_of_core(data_package):
	dp = data_package
	dest = 3
	comm.isend(rank,						dest=dest,	tag=5)
	comm.isend("notice_data_out_of_core",dest=dest,	tag=5)
	send_data_package(dp,	dest=dest,	tag=501)

def reader_give_access_to_data(data_name):
	dp = data_package_list[data_name]

	mem_inform(dp, 3)
	u = dp.unique_id

	mem_retain(dp)
	out_of_core = dp.out_of_core
	if out_of_core: 
		reader_notice_data_out_of_core(dp)
	else: 
		data = globals()[data_name]
		send_data(3, data, dp)

# memory manager related functions
##############################################################################
def mem_depth(unique_id, split_shape, split_position, depth):
	u = unique_id
	ss = split_shape
	sp = split_position

	comm.isend(rank,				dest=2,		tag=2)
	comm.isend("depth",			dest=2,		tag=2)
	comm.isend(u,				dest=2,		tag=24)
	comm.isend(ss,				dest=2,		tag=24)
	comm.isend(sp,				dest=2,		tag=24)
	comm.isend(depth,			dest=2,		tag=24)

def mem_inform(data_package, dest=None):
	dp = data_package
	comm.isend(rank,		dest=2,		tag=2)
	comm.isend("inform",	dest=2,		tag=2)

	temp1 = dp.data
	temp2 = dp.devptr
	dp.data = None
	dp.devptr = None

	send_data_package(dp, dest=2, tag=25)	
	dp.data = temp1
	dp.devptr = temp2

	comm.isend(dest,		dest=2,		tag=25)

def mem_method(u, m):
	comm.isend(rank,			dest=2,		tag=2)
	comm.isend("method",		dest=2,		tag=2)
	comm.isend(u,			dest=2,		tag=26)
	comm.isend(m,			dest=2,		tag=26)

def mem_release(data_package):
	comm.isend(rank,			dest=2,		tag=2)
	comm.isend("release",	dest=2,		tag=2)
	send_data_package(data_package, dest=2, tag=27)
		
def mem_retain(data_package):
	comm.isend(rank,			dest=2,		tag=2)
	comm.isend("retain",		dest=2,		tag=2)
	send_data_package(data_package, dest=2, tag=28)
	
def synchronize():
	comm.isend(rank,			dest=2,		tag=2)
	comm.isend("synchronize",dest=2,		tag=2)
	comm.recv(source=2, tag=999)

def VIVALDI_GATHER(data_package):
	dp = data_package
	if not isinstance(dp, Data_package): return dp

	u = dp.unique_id
	if u == None:
		return dp
	
	# ask gathering data 
	comm.isend(rank,			dest=2,		tag=2)
	comm.isend("gather",		dest=2,		tag=2)

	temp1 = dp.data
	temp2 = dp.devptr
	dp.data = None
	dp.devptr = None

	send_data_package(dp, dest=2, tag=212)
	dp.data = temp1
	dp.devptr = temp2

	# wati until data created, we don't know where the data will come from
	source = comm.recv(source=MPI.ANY_SOURCE, tag=5)
	flag = comm.recv(source=source, tag=5)

	task = comm.recv(source=source, tag=57)
	halo_size = comm.recv(source=source, tag=57)
	def recv():
		data_package = comm.recv(source=source, tag=52)
		dp = data_package
		data_memory_shape = dp.data_memory_shape
		
		dtype = dp.data_contents_memory_dtype
		data = numpy.empty(data_memory_shape, dtype=dtype)
		request = comm.Irecv(data, source=source, tag=57)
		MPI.Request.wait(request)
		
		return data, data_package
		
	data, data_package = recv()

	return data

# Reader related functions
def send_data(dest, data, data_package):
	dp = data_package

	comm.isend(rank,		dest=dest,	tag=5)
	comm.isend("recv",	dest=dest,	tag=5)
	
	t_data = dp.data
	t_devptr = dp.devptr
	dp.data = None
	dp.devptr = None

	send_data_package(dp, dest=dest, tag=52)
	dp.data = t_data
	dp.devptr = t_devptr
	t_data = None
	t_devptr = None

	if type(data) == numpy.ndarray:
		request = comm.Isend(data,		dest=dest,	tag=57)
		global requests
		requests.append((request, data))
#		MPI.Request.Wait(request)
	else:
		comm.isend(data,		dest=dest,	tag=52)

# GPU_unit related functions
######################################################################################
def register_function(execid, function_package):
	comm.isend(rank,				dest=2,		tag=2)
	comm.isend("function",			dest=2,		tag=2)
	comm.isend(function_package,	dest=2,		tag=22)

	#print vars(function_package)

# computing unit related functions
###############################################################################################
def isCPU(execid):
	if execid in computing_unit_list.keys():
		if computing_unit_list[execid]['device_type'] == 'CPU':return True
	return False

def isGPU(execid):
	if execid in computing_unit_list.keys():
		if computing_unit_list[execid]['device_type'] == 'GPU':return True
	return False

def get_any_CPU():
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'CPU':
			return i
	return None

def get_any_GPU():
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'GPU':
			return i
	return None

def get_another_CPU(execid):
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'CPU':
			if i == execid: continue
			return i
	return None

def get_another_GPU(execid):
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'GPU':
			if i == execid: continue
			return i
	return None

def get_CPU_list(number):
	rt_list = []
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'CPU':
			rt_list.append(i)
			number = number - 1
			if number == 0:
				return rt_list
	return rt_list

def get_GPU_list(number):
	rt_list = []
	cnt = 0
	for i in computing_unit_list:
		if computing_unit_list[i]['device_type'] == 'GPU':
			rt_list.append(i)
			number = number - 1
			cnt += 1
			if number == 0:
				return rt_list

	print "WARNING: NOT enough GPU only %d found"%(cnt)
	return rt_list

# load stream
##################################################################################################
def load_stream(file_full_name, dtype=None, size=32, halo=0):
	file_name, extension = split_file_name_and_extension(file_full_name)
	el = extension.lower()
	
	if el == 'dat':
		idx = file_full_name.rfind('dat')
		file_full_name = file_full_name[:idx] + 'str' + file_full_name[idx+3:]
		el = 'str'
		
		ss = read_file(file_full_name)
		ss = remove_comment(ss)
		info = to_dictionary(ss)
		os.system('./split %s %d %d'%(file_name, size, halo))
		
	if el == 'str':
		ss = read_file(file_full_name)
		file_number, file_dtype, file_shape, file_list = parse_str(ss)
		data_dtype = dtype if dtype != None else file_dtype
		
		ss = Shadie_Stream(file_list, file_dtype, file_shape, data_dtype)
		return ss
	
	print "Vivaldi error"
	print "this is not providing stream data type"
	assert(False)
	return {}

# save image
##################################################################################################
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

global cnt
cnt = 0
def get_file_name(file_name, data_name=None):
	if file_name == None:
		global cnt
		file_name = data_name + '_' +str(cnt)
		cnt += 1
		extension = 'png'
	else:
		file_name, extension = split_file_name_and_extension(file_name)

	return file_name, extension.lower()

def get_dimension_and_channel(input):

	shape = None
	if type(input) == numpy.ndarray:	shape = input.shape
	if type(input) == tuple:			shape = input
	shape = list(shape)

	n = len(shape)
	chan = shape[n-1]
	if chan in [2,3,4]:
		chan = chan
		shape.pop()
	else:
		chan = 1

	dimension = len(shape)
	return dimension, shape, chan

def save_data(input1, input2=None, out_of_core=False, noarmlize=False):
	save_image(input1, input2, out_of_core, normalize)

global file_name_cnt
file_name_cnt = 0	
def gen_file_name():
	global file_name_cnt
	file_name_cnt += 1
	return str(file_name_cnt)+'.png' # png is default extension

def save_image(input1, input2=None, out_of_core=False, normalize=False):
	dtype = 'float32'
	# merge image
	##########################################################################################################

	if input2 != None: # file name is exist
		data = input1
		file_name = input2
	else: # file name is not exist
		data = input1
		file_name = gen_file_name();

	flag = 0

	if isinstance(data, Data_package): flag = 1
	elif type(data) == numpy.ndarray:	flag = 2

	if flag == 1:
		data_name = data.data_name
		file_name, extension = get_file_name(file_name, data_name)
	
	if flag == 2:
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

	dp = data_package_list[data_name]
	u = dp.unique_id

	dp.file_name = file_name
	dp.extension = extension
	dp.normalize = normalize

	if out_of_core:
		dp.out_of_core = True
		
		u = data_list[data_name]
		for elem in retain_list[u]:
			dp.split_shape = elem.split_shape
			break
		reader_save_image_out_of_core(dp)
		VIVALDI_WRITE(data_name, None)
	else:
		reader_save_image_in_core(dp)

# etc
##################################################################################################
def strip_func(line):
	### <Summary> :: distinguish function name and function argument
	### Type of line :: str
	### if function not exist in this line
	### return False,"","",""
	### if function exist in this line
	### return True, func_name, func_args, after_func
	### ex) if(a<3):a=a+1
	### return True, if, a<3, :a=a+1
	tf = False
	f = line.find('(')
	if f != -1:
		name = line[:f]
		if name.strip().count(' ') == 0:pass
		else: return False, "", "", ""
		args = line[f+1: line.rfind(')')]
		after = line[line.rfind(')')+1:]
		return True, name, args, after
	return False, "", "", ""


# main
#####################################################################################################################
if __name__ == "__main__":
	if '-L' in sys.argv:
		idx = sys.argv.index('-L')
		log_type = sys.argv[idx+1]
	if '-G' in sys.argv:
		idx = sys.argv.index('-G')
		GPUDIRECT = sys.argv[idx+1]
		if GPUDIRECT == 'on':GPUDIRECT = True
		else: GPUDIRECT = False

	log("rank%d, Vivaldi main manager launch at %s"%(rank, name), "general", log_type)
	computing_unit_list = parent.recv(source=0, tag=1)
	global parsed_Vivaldi_functions
	parsed_Vivaldi_functions = parent.recv(source=0, tag=1)

	main_code = parsed_Vivaldi_functions.get_main()
	
	python_code_dict = parsed_Vivaldi_functions.get_python_code_dict()
	for function_name in python_code_dict:
		function_code = python_code_dict[function_name]
		if function_name == 'main':
			pass
		else:
			try:
				# developer should implement CPU version functions
				# than this part will be possible
				exec function_code
			except:
				print "Warning"
				print "--------------"
				print "Vivaldi Failed to \'exec\'"
				print function_code
				print "--------------"
				pass

	front = parsed_Vivaldi_functions.get_front()
	
	try:
		exec front
	except:
		print "Warning"
		print "--------------"
		print "Vivaldi Failed to \'exec\'"
		print front
		print "--------------"
		pass

	log('\n'+main_code,"parsing", log_type)
	# change main for use globals()
	##############################################################

	if log_type == 'parsing':
		print "FRONT"
		print "======================================"
		print front
		print "MAIN CODE"
		print "======================================"
		print main_code
	else:
	
		st = time.time()
		uchar = numpy.uint8
		
		try:
			exec main_code
		except:
			print "VIVALDI ERROR"
			print "---------------------"
			print "Problem during \'exec\' main_code"
			print main_code
			print "---------------------"
			assert(False)
		synchronize()
		print "TIME", time.time() - st

		if Vivaldi_viewer.viewer_on == True:
			Vivaldi_viewer.VIVALDI_GATHER = VIVALDI_GATHER
			Vivaldi_viewer.v.show()

		free_volumes()

	log("main function finished", "general", log_type)
	parent.send("main_finish",dest=0,tag=9)

	parent.Barrier()
	parent.Disconnect()
