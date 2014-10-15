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
		self.inv_mmtx = numpy.eye(4,dtype=numpy.float32)

		self.trans_tex = numpy.ndarray(256*4,dtype=numpy.int8)
		self.update_tf = 0
		self.update_tf2 = 0
		self.trans_tex = numpy.ndarray(256*4, dtype=numpy.uint8)
		self.TF_bandwidth = 1
		self.CRG = 0

	def __str__(self):
		### print Vivaldi_code parsed result
		#####################################################################
		sbuf = ''
		sbuf += "function package\n"
		sbuf += "--------------------------------------------\n"
		sbuf += "function_name  : "+str(self.function_name)+'\n'
		sbuf += "work_range     : "+str(self.work_range)+'\n'
		sbuf += "mmtx			: \n"+str(self.mmtx)+'\n'
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
		
	def copy(self):
		temp = copy.copy(self)
		temp.output = copy.copy(self.output)
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
	def __init__(self, data_range=None):
		self.unique_id = None
		self.split_shape = str(SPLIT_BASE)
		self.split_position = str(SPLIT_BASE)
		self.data_name = None
		self.memory_type = None
		self.merge_method = None
		self.depth = 0
		self.retain_counter = 1
		self.usage = 0
		
		# data
		self.data_range = dict()
		self.data_halo = 0
		self.data_dtype = None
		self.data_shape = None
		self.data_memory_shape = None
		self.data_bytes = None
		self.data_contents_dtype = 'float'
		self.data_contents_memory_dtype = numpy.float32
		self.data_contents_memory_shape = [1]
		
		# full data
		self.full_data_range = dict()
		self.full_data_shape = None
		self.full_data_memory_shape = None
		self.full_data_bytes = None

		# buffer 
		self.buffer_range = dict()
		self.buffer_halo = 0
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
		halo = 0
		if type(input) == str:
			input = ast.literal_eval(input)
		if 'halo' in input:
			halo = input['halo']
			del(input['halo'])

		self.data_range = apply_halo(input, halo)

		shape = range_to_shape(self.data_range)
		memory_shape = make_memory_shape(shape, self.data_contents_memory_shape)

		bcmd = self.data_contents_memory_dtype
		bcmb = get_bytes(bcmd)

		bytes = make_bytes(memory_shape, bcmb)

		if bytes < 0:
			print "VIVALDI BUG"
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
		pass

	def set_full_data_range(self, input):
		halo = 0
		if type(input) == str:
			input = ast.literal_eval(input)

		if 'halo' in input:
			halo = input['halo']
			del(input['halo'])
		self.full_data_range = apply_halo(input, halo)

		shape = range_to_shape(self.full_data_range)
		memory_shape = make_memory_shape(shape, self.data_contents_memory_shape)
		
		bcmd = self.data_contents_memory_dtype
		bcmb = get_bytes(bcmd)

		bytes = make_bytes(memory_shape, bcmb)
		
		self.full_data_shape = shape
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
			self.buffer_range = apply_halo(self.data_range, -data_halo+buffer_halo)
		
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
		if a == None:return "%s\t\t\tNone\n"%(name)
		return "%s\t\t\t%s\n"%(name,str(a))

	def __str__(self):
		### print Vivaldi_code parsed result
		#####################################################################
		sbuf = ''
		sbuf += "Data package\n"
		sbuf += "--------------------------------------------\n"
		sbuf += self.pt("unique_id*\t\t\t\t", self.unique_id)
		sbuf += self.pt("split_shape*\t\t\t\t", self.split_shape)
		sbuf += self.pt("split_position*\t\t\t", self.split_position)
		sbuf += self.pt("data_name\t\t\t\t", self.data_name)
		sbuf += self.pt("memory_type\t\t\t\t", self.memory_type)
	
		sbuf += self.pt("data_range\t\t\t", self.data_range)
		sbuf += self.pt("data_halo\t\t\t", self.data_halo)
		sbuf += self.pt("data_shape\t\t\t", self.data_shape)
		sbuf += self.pt("data_memory_shape\t\t", self.data_memory_shape)
		sbuf += self.pt("data_dtype\t\t\t",self.data_dtype)	
		sbuf += self.pt("data_contents_dtype\t",self.data_contents_dtype)
		sbuf += self.pt("data_contents_memory_dtype", self.data_contents_memory_dtype)
		sbuf += self.pt("data_contents_memory_shape", self.data_contents_memory_shape)
		sbuf += self.pt("data_bytes\t\t\t\t\t",self.data_bytes)
	

		sbuf += self.pt("full_data_range\t\t\t",self.full_data_range)
		sbuf += self.pt("full_data_shape\t\t",self.full_data_shape)
		sbuf += self.pt("full_data_memory_shape",self.full_data_memory_shape)
		sbuf += self.pt("full_data_bytes\t\t\t\t\t",self.full_data_bytes)

		sbuf += self.pt("buffer_range\t\t\t", self.buffer_range)
		sbuf += self.pt("buffer_halo\t\t\t\t", self.buffer_halo)
		sbuf += self.pt("buffer_bytes\t\t\t\t\t",self.buffer_bytes)
		
		sbuf += self.pt("depth\t\t\t\t\t",self.depth)
		sbuf += self.pt("data\t\t\t\t\t",self.data)
		sbuf += self.pt("devptr\t\t\t\t\t",self.devptr)


		sbuf += self.pt("out_of_core\t\t\t", self.out_of_core)
		sbuf += self.pt("file_name\t\t\t", self.file_name)
		sbuf += self.pt("extension\t\t\t", self.extension)
		sbuf += self.pt("file_dtype\t\t\t", self.file_dtype)

		sbuf += "--------------------------------------------\n"
		return sbuf

	def copy(self):
		temp = copy.copy(self)
		temp.buffer_halo = self.buffer_halo
		temp.split_shape = self.split_shape
		temp.split_position = self.split_position
		temp.data_name = self.data_name
		temp.memory_type = self.memory_type
		temp.merge_method = self.merge_method
		temp.depth = self.depth
		
		temp.full_data_range = dict(self.full_data_range)
		temp.full_data_shape = self.full_data_shape
		temp.full_data_memory_shape = self.full_data_memory_shape
		temp.full_data_bytes = self.full_data_bytes
		
		temp.data_range = dict(self.data_range)
		temp.data_halo = self.data_halo
		temp.data_dtype = self.data_dtype
		temp.data_shape = self.data_shape
		temp.data_memory_shape = self.data_memory_shape
		temp.data_bytes = self.data_bytes
		temp.data_contents_dtype = self.data_contents_dtype
		temp.data_contents_memory_dtype = self.data_contents_memory_dtype
		temp.data_contents_memory_shape = list(self.data_contents_memory_shape)

		temp.out_of_core = self.out_of_core
		temp.file_name	= self.file_name
		temp.file_dtype = self.file_dtype
		temp.extension = self.extension
		temp.normalize = self.normalize

		temp.data = None
		temp.devptr = None
		return temp
