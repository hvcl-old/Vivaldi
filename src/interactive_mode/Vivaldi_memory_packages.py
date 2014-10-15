from Vivaldi_misc import *


class Function_package():
	def __init__(self):
		self.function_name = None
		self.function_args = None
		self.work_range = None
		self.execid_list = []
		self.reserved = False
		self.output = Data_package()
		self.mmtx = numpy.eye(4,dtype=numpy.float32)
		#self.inv_mmtx = numpy.eye(4,dtype=numpy.float32)
		
		self.transN = 0
		self.trans_tex = None
		self.update_tf = 0
		self.update_tf2 = 0
		self.TF_bandwidth = 1
		self.Sliders = numpy.zeros(4, dtype=numpy.int32)
		self.Slider_opacity = numpy.zeros(4, dtype=numpy.int32)
		
	def info(self):
		### print Vivaldi_code parsed result
		#####################################################################
		sbuf = ''
		sbuf += "function package\n"
		sbuf += "--------------------------------------------\n"
		sbuf += "function_name  : "+str(self.function_name)+'\n'
#		sbuf += "function_args  : "
#		for elem in self.function_args:
#			if isinstance(elem, Data_package):
#				sbuf += elem.data_name + ':' + str(elem.data_contents_dtype) + ', '
#		sbuf += '\n'
		sbuf += "work_range     : "+str(self.work_range)+'\n'
		sbuf += "mmtx           : \n"+str(self.mmtx)+'\n'
		sbuf += "--------------------------------------------\n"
		return sbuf
	# setter check data type is valid or not
	def set_function_name(self, function_name):
		if type(function_name) == str:
			self.function_name = function_name
		else:
			print "Invalid function_name data type"
			print function_name
			assert(False)
	def set_function_args(self, argument_package_list):
		self.function_args = argument_package_list
	def get_function_args(self):
		return self.function_args
	def set_args(self, argument_package_list):
		self.function_args = argument_package_list
	def get_args(self):
		return self.function_args
		
	def copy(self):
		temp = copy.deepcopy(self)
		return temp
	
class Memcpy_task():
	def __init__(self):
		self.source = Data_package()
		self.dest = Data_package()
		self.work_range = None
		self.func_name = None
		self.execid = None
		self.start = None

class Data_package():
	def __init__(self, data=None, halo=0, split=None):
		def default(self):
			self.unique_id = None
			self.split_shape = str(SPLIT_BASE)
			self.split_position = str(SPLIT_BASE)
			self.data_name = None
			self.memory_type = None
			self.merge_method = None
			self.depth = None
			self.retain_counter = 1
			self.usage = None
			self.shared = False
			
			# data
			self.data_range = None
			self.data_halo = None
			self.data_dtype = None
			self.data_shape = None
			self.data_memory_shape = None
			self.data_bytes = None
			self.data_contents_dtype = None
			self.data_contents_memory_dtype = None
			self.data_contents_memory_shape = None
			
			# full data
			self.full_data_range = None
			self.full_data_shape = None
			self.full_data_memory_shape = None
			self.full_data_bytes = None
			self.data_contents_memory_shape = None
			self.data_contents_memory_shape = None
			
			# buffer 
			self.buffer_range = None
			self.buffer_halo = None
			self.buffer_bytes = None
			
			self.out_of_core = False
			self.file_name = None
			self.file_dtype = None
			self.extension = None
			self.normalize = False

			self.data = None
			self.devptr = None
			self.hard_disk = None

			self.x = 0
			self.y = 0
			self.z = 0
		default(self)
		if data == None:
			pass
		elif isinstance(data, Data_package):
			self = data.copy()
			
		elif data in AXIS:
			# AXIS symbols
			def set_AXIS_package(axis_name):
				self.data_name = axis_name
				self.unique_id = -1
				self.data_dtype = int
				self.data_contents_dtype = 'int'
				self.data_contents_memory_dtype = numpy.int32
				self.data = None		
			set_AXIS_package(axis_name=data)
		else: # except Data_package and AXIS
			if type(data) == numpy.ndarray: # numpy 
				def set_volume_package(volume):
					self.data_dtype = numpy.ndarray
					self.data = data
					def set_full_data_range(data):
						def get_data_contents_dtype(data):
							shape = list(data.shape)
							type_name = python_dtype_to_Vivaldi_dtype(data.dtype) # Vivaldi misc
							last = list(data.shape)[-1]
							if last in [2,3,4]:
								chan = [last]
								type_name += str(last)
								shape.pop()
							
							return type_name, shape
						def get_data_contents_memory_dtype(data):
							return data.dtype
						
						self.data_contents_dtype, shape = get_data_contents_dtype(data)
						self.set_data_contents_dtype(self.data_contents_dtype)
						
						self.full_data_shape = shape
						def shape_to_range(shape):
							full_data_range = {}
							n = len(shape)
							if n >= 1:full_data_range['x'] = (0, shape[n-1])
							if n >= 2:full_data_range['y'] = (0, shape[n-2])
							if n >= 3:full_data_range['z'] = (0, shape[n-3])
							if n >= 4:full_data_range['w'] = (0, shape[n-4])
							return full_data_range
						self.full_data_range = shape_to_range(shape)
						self.full_data_memory_shape = data.shape
						self.full_data_bytes = data.nbytes
						
					set_full_data_range(data)
					self.set_data_range(self.full_data_range)
				set_volume_package(volume=data)
			else: # ordinary variables
				def set_value_package(value):
					self.unique_id = -1
					self.data_dtype = type(value)
					self.data_contents_dtype = python_dtype_to_Vivaldi_dtype(type(value))
					self.data = value
				set_value_package(value=data)
	
		# modfiier
		self.halo = halo
		self.data_halo = halo
		self.split = split
	
	def get_unique_id(self):
		return str(self.unique_id)
	def get_split_shape(self):
		return str(self.split_shape)
	def get_split_position(self):
		return str(self.split_position)
	def get_id(self):
		return (self.get_unique_id(), self.get_split_shape(), self.get_split_position())
	def split_data_range(self, split_position):
		new_range = {}
		split_shape = eval(self.split_shape)
		for axis in AXIS:
			if axis in self.full_data_range:
				w = self.full_data_range[axis][1] - self.full_data_range[axis][0]
				n = split_shape[axis] if axis in split_shape else 1
				full_start = self.full_data_range[axis][0]
				m = split_position[axis]-1
				
				start = m*w/n+full_start-self.data_halo
				if start < self.full_data_range[axis][0]: start = self.full_data_range[axis][0]
				end = (m+1)*w/n+full_start+self.data_halo
				if end > self.full_data_range[axis][1]: end = self.full_data_range[axis][1]
				
				new_range[axis] = (start, end)
		
		self.set_data_range(new_range)
	def set_usage(self, usage):
		if usage == 0:
			print "VIVALDI system error"
			print "----------------------------------"
			print "Wrong memory usage:", usage
			print "----------------------------------"
			assert(False)
		self.usage = usage
	def get_bytes(self):
		if self.data_bytes == None: return self.full_data_bytes
		return self.data_bytes
	def set_dtype(self, dtype):
		self.set_data_contents_dtype(dtype)
		self.set_data_range(self.data_range)
		self.set_full_data_range(self.full_data_range)
		
	def set_data_contents_dtype(self, dtype):
		dtype = python_dtype_to_Vivaldi_dtype(dtype)
		ms = [1]
		md = Vivaldi_dtype_to_python_dtype(dtype)
		
		if type(dtype) == numpy.dtype: pass
		elif dtype in ['uchar2','int2','short2','float2','double2']:ms = [2]
		elif dtype in ['RGB','uchar3','short3','int3','float3','double3']:ms = [3]
		elif dtype in ['RGBA','uchar4','short4','int4','float4','double4']: ms = [4]
	
		self.data_contents_dtype = dtype
		self.data_contents_memory_dtype = md
		self.data_contents_memory_shape = ms
	
	def set_data_range(self, input):
		if type(input) == str:
			input = ast.literal_eval(input)
		self.data_range = input
		shape = range_to_shape(self.data_range)
		memory_shape = make_memory_shape(shape, self.data_contents_memory_shape)
		bcmd = self.data_contents_memory_dtype
		bcmb = get_bytes(bcmd)
		bytes = make_bytes(memory_shape, bcmb)
		if bytes < 0:
			print "VIVALDI system error"
			print "----------------------------------"
			print "Wrong bytes size: ", bytes
			print "Related variables"
			print "Memory shape:",memory_shape
			print "Byte:",bcmb
			print "Memory type:",bcmd
			print "----------------------------------"
			assert(False)
		self.data_shape = shape
		self.data_bytes = int(bytes)
		self.data_memory_shape = memory_shape
		
	def set_full_data_range(self, full_data_range):
		if type(full_data_range) == str:
			full_data_range = ast.literal_eval(full_data_range)

		self.full_data_range = full_data_range
		shape = range_to_shape(self.full_data_range)
		self.full_data_shape = shape
		
		memory_shape = make_memory_shape(shape, self.data_contents_memory_shape)
		
		bcmd = self.data_contents_memory_dtype
		
		bcmb = get_bytes(bcmd)
		bytes = make_bytes(memory_shape, bcmb)
		
		self.full_data_bytes = int(bytes)
		self.full_data_memory_shape = memory_shape


		if len(shape) == 3:
			self.z, self.y, self.x = shape
		if len(shape) == 2:
			self.y, self.x = shape
		if len(shape) == 1:
			self.x = shape[0]
		pass

	def set_buffer_range(self, input):
		flag = 0
		# preprocessing
		if type(input) == int:		
			flag = 1
		elif type(input) == str:
			input = ast.literal_eval(input)
		
		# set buffer range
		if flag == 0:			
			self.buffer_range = dict(input)
		elif flag == 1:			
			data_halo = self.data_halo
			buffer_halo = input
			self.buffer_halo = buffer_halo
			self.buffer_range = apply_halo(self.data_range, -data_halo+buffer_halo, self.full_data_range)
		
		# related values
		shape = range_to_shape(self.buffer_range)
		memory_shape = make_memory_shape(shape, self.data_contents_memory_shape)
		bcmd = self.data_contents_memory_dtype
		bcmb = get_bytes(bcmd)
		bytes = make_bytes(memory_shape, bcmb)
		self.buffer_bytes = int(bytes)
		
	def set_halo(self, new_halo):
		old_halo = self.buffer_halo
		data_range = self.data_range

		data_range = apply_halo(data_range, new_halo-old_halo)

		full_data_range = self.full_data_range
		full_data_range = apply_halo(full_data_range, new_halo-old_halo)

		self.set_data_range(data_range)
		self.set_full_data_range(full_data_range)

		self.buffer_halo = new_halo

	def pt(self, name, a):
		#if a == None:return "%s\t\t\tNone\n"%(name)
		if a == None: return ''
		return "%s%s\n"%(name,str(a))

	def info(self):
		### print Vivaldi_code parsed result
		#####################################################################
		sbuf = ''
		sbuf += "Data package\n"
		sbuf += "--------------------------------------------\n"
		sbuf += self.pt("unique_id*                         ", self.unique_id)
		sbuf += self.pt("split_shape*                       ", self.split_shape)
		sbuf += self.pt("split_position*                    ", self.split_position)
		sbuf += self.pt("data_name                          ", self.data_name)
		sbuf += self.pt("memory_type                        ", self.memory_type)
	
		sbuf += self.pt("data_range                         ", self.data_range)
		sbuf += self.pt("data_halo                          ", self.data_halo)
		sbuf += self.pt("data_shape                         ", self.data_shape)
		sbuf += self.pt("data_memory_shape                  ", self.data_memory_shape)
		sbuf += self.pt("data_dtype                         ", self.data_dtype)	
		sbuf += self.pt("data_contents_dtype                ", self.data_contents_dtype)
		sbuf += self.pt("data_contents_memory_dtype         ", self.data_contents_memory_dtype)
		sbuf += self.pt("data_contents_memory_shape         ", self.data_contents_memory_shape)
		sbuf += self.pt("data_bytes                         ", self.data_bytes)
	

		sbuf += self.pt("full_data_range                    ", self.full_data_range)
		sbuf += self.pt("full_data_shape                    ", self.full_data_shape)
		sbuf += self.pt("full_data_memory_shape             ", self.full_data_memory_shape)
		sbuf += self.pt("full_data_bytes                    ", self.full_data_bytes)

		sbuf += self.pt("buffer_range                       ", self.buffer_range)
		sbuf += self.pt("buffer_halo                        ", self.buffer_halo)
		sbuf += self.pt("buffer_bytes                       ", self.buffer_bytes)
		sbuf += self.pt("usage                              ", self.usage)
		
		sbuf += self.pt("depth                              ", self.depth)
		sbuf += self.pt("data\n",self.data)
		sbuf += self.pt("devptr                             ", self.devptr)


		sbuf += self.pt("out_of_core                        ", self.out_of_core)
		sbuf += self.pt("file_name                          ", self.file_name)
		sbuf += self.pt("extension                          ", self.extension)
		sbuf += self.pt("file_dtype                         ", self.file_dtype)

		#sbuf += self.pt("modifier, halo                     ", self.halo)
		#sbuf += self.pt("modifier, split                    ", self.split)

		sbuf += "--------------------------------------------\n"
		return sbuf

	def copy(self):
		t_data, t_devptr = self.data, self.devptr
		self.data, self.devptr = None, None
		temp = copy.deepcopy(self)
		self.data, self.devptr = t_data, t_devptr
		t_data, t_devptr = None, None
		return temp
