import matrixio
import Image
import time
from Vivaldi_misc import *

from Vivaldi_memory_packages import Data_package

# def load_data_1d(filename, name, mod):
#     data = read_1d_data(filename)
#     texref = create_1d_texture(data, mod, name)
#     return texref, data.shape


# def load_data_2d(filename, name, mod):
#     data = read_2d_data(filename)
#     texref = create_2d_texture(data, mod, name)
#     return texref, data.shape


def load_data_2d(file_full_name, dtype=None, out_of_core=False):
	dtype = python_dtype_to_Vivaldi_dtype(dtype)
	file_name, extension = split_file_name_and_extension(file_full_name)
	el = extension.lower()
	if out_of_core:
		if extension == 'dat':
			file_info = matrixio.read_dat(filename, out_of_core=True)
			file_name = file_info['FileName']
			assert(file_name != None)
			
			if 'Path' in file_info:
				# user defined path
				path = file_info['Path']
			else:
				# default is find same folder
				idx = file_full_name.rfind('/')
				path = file_full_name[:idx+1] 

			shape = file_info['Resolution'].split()
			shape[0],shape[1] = int(shape[1]),int(shape[0])
			contents_dtype = file_info['Format'].lower()
			chan = 1

			temp = Data_package()
			temp.out_of_core = True

			temp.file_name = path + file_name
			temp.file_dtype = python_dtype_to_Vivaldi_dtype(contents_dtype)

			temp.buffer_dtype = numpy.ndarray
			if dtype != None: dtype = dtype
			else: dtype = contents_dtype
			if chan != 1: dtype += str(chan)
	
			temp.set_buffer_contents_dtype(dtype)
	
			data_range = shape_to_range(shape)
			temp.set_data_range(data_range)
			temp.set_buffer_range(data_range)
			temp.set_full_buffer_range(data_range)

			return temp

	else:
		st = time.time()
		data = matrixio.read_2d_data(file_full_name)

		dtype = Vivaldi_dtype_to_python_dtype(dtype)
		if dtype != None:
			data = data.astype(dtype)
		else: dtype = data.dtype
		print 'load_data_2d_rgb:', file_full_name, 'dtype:', data.dtype, 'shape:', data.shape,'loading time:',1000*(time.time()-st),'ms'
	
		return data
	return None


def load_data_3d(file_full_name, dtype=None, out_of_core=False):
	dtype = python_dtype_to_Vivaldi_dtype(dtype)

	file_name, extension = split_file_name_and_extension(file_full_name)
	el = extension.lower()

	if out_of_core:
		if el == 'dat':
			# read dat file
			# dat file have FileName and 
			# It may have AbsolutePath or RelativePath
			file_info = matrixio.read_dat(file_full_name, out_of_core=True)
			file_name = file_info['FileName']
			assert(file_name != None)
			
			if 'Path' in file_info:
				# user defined path
				path = file_info['Path']
			else:
				# default is find same folder
				idx = file_full_name.rfind('/')
				path = file_full_name[:idx+1] 
				

			shape = file_info['Resolution'].split()
			shape[0],shape[1],shape[2] = int(shape[2]),int(shape[1]),int(shape[0])
			contents_dtype = file_info['Format'].lower()
			chan = file_info['Channel'] if 'Channel' in file_info else 1

			temp = Data_package()
			temp.out_of_core = True

			temp.file_name = path + file_name
			temp.file_dtype = python_dtype_to_Vivaldi_dtype(contents_dtype)

			temp.buffer_dtype = numpy.ndarray
			if dtype != None: dtype = dtype
			else: dtype = contents_dtype
			if chan != 1: dtype += str(chan)

			temp.set_data_contents_dtype(dtype)
	
			data_range = shape_to_range(shape)

			temp.set_data_range(data_range)
			temp.set_buffer_range(data_range)
			temp.set_full_data_range(data_range)
			return temp

		else:
			print "NOT PROVIDE DATA TYPE"
			assert(False)
	else:
		st = time.time()
		data = matrixio.read_3d_data(file_full_name)

		if dtype != None:
			dtype = python_dtype_to_Vivaldi_dtype(dtype)
			dtype = Vivaldi_dtype_to_python_dtype(dtype)
			data = data.astype(dtype)

		if data == None:
			print "======================================"
			print "VIVALDI ERROR, data load is failed"
			print "Are you sure, exension is correct?"
			print "Extension: %s"%(extension)
			print "======================================"
			assert(False)



		print 'load_data_3d:', file_full_name, 'dtype:', data.dtype, 'shape:', data.shape,'loading time:',1000*(time.time()-st),'ms'
		return data

	return None

def load_data_4d(filename, dtype):
	data = matrixio.read_4d_data(filename)
	print 'load_data_4d:', filename, 'dtype:', data.dtype, 'shape:', data.shape
	return data
	pass

def to_dictionary(ss):
	return_dict = {}
	for line in ss:
		pos = line.find(':')
		if pos == -1:continue
		key = line[:pos].strip()
		value = line[pos+1:].strip()
		return_dict[key] = value
	return return_dict

def parse_str(ss):
	ss = remove_comment(ss,'#')
	s_ss = ss.split('\n')

	# file number and dtype is in the first line 
	# in number, dtype order
	f_line = s_ss[0].split(' ')

	# file number
	file_number = int(f_line[0].strip())
	# file dtype
	file_dtype = f_line[1].strip()
	file_dtype = Vivaldi_dtype_to_python_dtype(file_dtype)

	# file shap

	shape = []
	for elem in s_ss[1].strip().split(' '):
		shape.append(int(elem.strip()))

	file_shape = tuple(shape)

	# file list 
	file_list = s_ss[2:]

	return file_number, file_dtype, file_shape,  file_list

def to_tuple(ss):
	return_tuple = []
	for elem in ss.split():
		elem = elem.strip()
		return_tuple.append(elem)

	return tuple(return_tuple)
