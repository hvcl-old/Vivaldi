# small helper functions go here
#common import
import time, numpy, sys, copy, ast, os


GIGA = float(1024*1024*1024)
MEGA = float(1024*1024)
KILO = float(1024)
AXIS = ['x','y','z','w']
SPLIT_BASE = {'x':1,'y':1,'z':1,'w':1}
modifier_list = ['execid','split','range','merge','halo','dtype','modifier']

log_type = ""
GPUDIRECT = "on"

BLOCK_SIZE = 16

OPERATORS = ["==",'+=','-=','*=','/=','=','+','-','*','/',"<=","=>","<",">","."]

def log(a, ptype, type):
	# log print type
	# time
	# parsing
	# general
	# detail
	# all
	flag = False
	if ptype == 'all' or type == 'all':
		flag = True

	if type == 'detail' and ptype in ['detail', 'general']:
		flag = True

	if type == ptype:
		flag = True
	
	if flag == True:
		print "[%.2f] %s"%(time.time(), a)

def read_file(name):
	"read a file into a string"
	f = open(name)
	x = f.read()
	f.close()
	return x
    
    
def write_file(name, x):
    "write string into a file"
    f = open(name, "w")
    f.write(x)
    f.close()
    
    
def normalize(v):
    return v / sqrt(dot(v, v))
    

def cross(a, b):
    return array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


def mk_frame(position, target, up):
    "create camera frame"
    
    z = normalize(target - position)
    x = normalize(cross(z, up))
    y = normalize(cross(z, x))
    
    return r_[x, y, z, position]


def find_index(p, l):
    "find first index in l that satisfies predicate p"
    for i, x in enumerate(l):
        if p(x):
            return i
    return None
    
    
def rotation_matrix(axis, angle):
    x, y, z = axis
    c = cos(angle)
    u_cross = r_[0, -z, y, z, 0, -x, -y, x, 0]
    u_cross.shape = 3, 3
    return c * eye(3) + sin(angle) * u_cross + (1-c) * outer(axis, axis)



# etc
##########################################################################################################################

def make_bytes(in1=None, in2=None, in3=None):
	bytes = 1
	temp_list = [in1,in2,in3]

	for elem in temp_list:
		if elem == None:
			pass
		elif type(elem) in [tuple,list]:
			for x in elem: bytes *= x
		elif type(elem) == dict:
			for x in elem: bytes *= elem[x][1]-elem[x][0]
		else:
			bytes *= elem

	return bytes


def make_split_position(split_shape, n):
	if n == 0:
		print "I assumed n start from 1. if you want to set n as zero, please double check"

	p = 0
	split_position = dict(SPLIT_BASE)
	ss = split_shape
	sp = split_position
	sp[AXIS[0]] = n
	for axis in AXIS:
		if axis not in split_shape:
			split_shape[axis] = 1


	for axis in AXIS[0:-1]:
		"""
		while sp[AXIS[p]] > ss[AXIS[p]]:
			sp[AXIS[p]] -= ss[AXIS[p]]
			sp[AXIS[p+1]] += 1
		p += 1
		"""

		q = sp[AXIS[p]]
		w = ss[AXIS[p]]
		a = int(q/w)

		if a*w == q: a -= 1
		sp[AXIS[p]] -= a*w
		sp[AXIS[p+1]] += a
		p += 1

	return split_position
		
		
def shape_to_range(shape):
	full_data_range = {}
	n = len(shape)
	if n >= 1:full_data_range['x'] = (0, shape[n-1])
	if n >= 2:full_data_range['y'] = (0, shape[n-2])
	if n >= 3:full_data_range['z'] = (0, shape[n-3])
	if n >= 4:full_data_range['w'] = (0, shape[n-4])

	return full_data_range
		


def range_to_shape(rg):
	temp = []
	for axis in AXIS[::-1]:
		if axis in rg:
			temp.append(rg[axis][1]-rg[axis][0])
	return tuple(temp)


def shape_to_count(shape):
	count = 1
	for num in shape:
		count *= shape[num]
	return count


def make_cuda_list(rg, size=-1, order=False):
	temp = []
	if size == -1: size = len(rg)
	if order == True: LT = AXIS[0:size]
	else: LT = AXIS[0:size][::-1]
	for axis in LT:
		if axis in rg:
			temp.append(numpy.int32(rg[axis][0]))
			temp.append(numpy.int32(rg[axis][1]))
		else:
			temp.append(numpy.int32(0))
			temp.append(numpy.int32(0))
	return temp

def range_to_block_grid(rg, size=8, div=1):
	block = []
	grid = []
	for axis in AXIS[:3]:
		if axis in rg:
			block.append(int(size))
			grid.append((rg[axis][1] - rg[axis][0]-1)/size + 1)
		else:
			block.append(int(1))
			grid.append(int(1))
	block = tuple(block)
	grid = tuple(grid)
	return block, grid

def make_memory_shape(shape, cms):
	memory_shape = tuple(shape)
	if cms != [1]: memory_shape = tuple(list(shape) + cms)
	return tuple(memory_shape)

def make_range_list(full_range, split, halo=0):
	# make data range list
	range_list = []
	for axis in AXIS:
		if axis in full_range:
			temp = []
			w = full_range[axis][1] - full_range[axis][0]
			n = split[axis] if axis in split else 1
			start = full_range[axis][0]
			for m in range(n): temp.append( {axis:(m*w/n+start-halo, (m+1)*w/n+start+halo)} )
				
			if len(range_list) == 0:
				for elem in temp: range_list.append(elem)
			else:
				temp2 = []
				for elem2 in temp:
					for elem1 in range_list:
						t = {}
						t.update(elem1)
						t.update(elem2)
						temp2.append(t)
				range_list = temp2

	return range_list

def make_depth(data_range,mmtx):
	x,y,z,depth = 0, 0, 0, 0
	for axis in data_range:
		if axis in ['x','y','z']:
			if axis == 'x':x = (data_range[axis][0] + data_range[axis][1])/2
			if axis == 'y':y = (data_range[axis][0] + data_range[axis][1])/2
			if axis == 'z':z = (data_range[axis][0] + data_range[axis][1])/2
		
	#X = mmtx[0][0]*x+mmtx[0][1]*y+mmtx[0][2]*z+mmtx[0][3]
	#Y = mmtx[1][0]*x+mmtx[1][1]*y+mmtx[1][2]*z+mmtx[1][3]
	Z = mmtx[2][0]*x+mmtx[2][1]*y+mmtx[2][2]*z+mmtx[2][3]
	depth = Z*Z
	return depth

def dtype_to_contents_memory_shape(dtype=None):
	data_shape = [1]
	if dtype == 'Rgba':data_shape = [4]
	if dtype == 'Rgb':data_shape = [3]
	if dtype == 'float4':data_shape = [4]
	if dtype == 'float3':data_shape = [3]
	if dtype == 'float2':data_shape = [2]
	return data_shape


def shape_to_int3(ttt):
	outDim = numpy.zeros((3), dtype=numpy.int32)
	if len(ttt) == 3:
		outDim[0] = numpy.int32(ttt[2])
		outDim[1] = numpy.int32(ttt[1])
		outDim[2] = numpy.int32(ttt[0])
		
	if len(ttt) == 2:
		outDim[0] = numpy.int32(ttt[1])
		outDim[1] = numpy.int32(ttt[0])

	if len(ttt) == 1:
		outDim[0] = numpy.int32(ttt[0])
		outDim[1] = numpy.int32(0)

	return outDim

def get_start_end(rg, halo=0):
	s = numpy.zeros((3), dtype=numpy.int32)
	e = numpy.zeros((3), dtype=numpy.int32)

	if type(rg) == 'str': rg = ast.literal_eval(rg)
	i = 0
	for axis in AXIS:
		if axis in rg:
			s[i] = rg[axis][0] + halo
			e[i] = rg[axis][1] - halo
		i += 1
	return s,e




def make_func_name_with_dtypes(name, func_args, dtypes):
	
	new_name = str(name)

	for data_name in func_args:
		if data_name in dtypes:
			new_name += data_name.replace('.','_') + dtypes[data_name].replace('.','_')
		
	return new_name

def buffer_range_minus_halo(buffer_range, halo):
	temp = {}
	for axis in AXIS:
		if axis in buffer_range:
			temp[axis] = (buffer_range[axis][0]+halo, buffer_range[axis][1]-halo)
			
	return temp

def apply_halo(buffer_range, halo, full_buffer_range=None):


	if full_buffer_range == None:
		temp = {}
		for axis in AXIS:
			if axis in buffer_range:
				temp[axis] = (buffer_range[axis][0]-halo, buffer_range[axis][1]+halo)
	else:
		temp = {}
		for axis in AXIS:
			if axis in buffer_range:

				start = buffer_range[axis][0]-halo
				end = buffer_range[axis][1]+halo

				start = full_buffer_range[axis][0] if start < full_buffer_range[axis][0] else start
				end = full_buffer_range[axis][1] if full_buffer_range[axis][1] < end else end

				temp[axis] = (start, end)
	

	return temp

def python_dtype_to_bytes(python_dtype):
	bcmb = -1
	bytes = bcmb
	if python_dtype in [numpy.int8, numpy.uint8]: bytes = 1
	if python_dtype in [numpy.int16, numpy.uint16]: bytes = 2
	if python_dtype in [numpy.int32, numpy.uint32, int]: bytes = 4
	if python_dtype in [numpy.float32, float]: bytes = 4
	if python_dtype in [numpy.float64]: bytes = 8

	return bytes

def vivaldi_dtype_to_bytes(vivaldi_dtype):
	bytes = -1
	try:
		if vivaldi_dtype == 'uchar': bytes = 1
		if vivaldi_dtype == 'short': bytes = 2
		if vivaldi_dtype == 'int': bytes = 4
		if vivaldi_dtype == 'float': bytes = 4
		if vivaldi_dtype == 'double': bytes = 8

		if vivaldi_dtype in ['RGB','RGBA','uchar','uchar2','uchar3','uchar4']: bytes = 1
		if vivaldi_dtype in ['short','short2','short3','short4']: bytes = 2
		if vivaldi_dtype in ['int','int2','int3','int4','float','float2','float3','float4']: bytes = 4
	
		if bytes == -1:
			# not vivaldi dtype
			pass
	except:
		pass
	return bytes


def get_bytes(type):
	
	bytes = vivaldi_dtype_to_bytes(type)
	if bytes == -1:
		bytes = python_dtype_to_bytes(type)
	
	return bytes


def vivaldi_dtype_to_python_dtype(vivaldi_dtype):
	python_dtype = None
	if vivaldi_dtype in ['uchar','uchar2','uchar3','uchar4']: python_dtype = numpy.uint8
	if vivaldi_dtype in ['short','short2','short3','short4']: python_dtype = numpy.uint16
	if vivaldi_dtype in ['int','int2','int3','int4']  : python_dtype = numpy.int32
	if vivaldi_dtype in ['float','float2','float3','float4']: python_dtype = numpy.float32
	if vivaldi_dtype in ['double','double2','double3','double4']: python_dtype = numpy.float64

	if vivaldi_dtype in ['RGB','RGBA']: python_dtype = numpy.uint8

	if type(vivaldi_dtype) != str: python_dtype = vivaldi_dtype
	return python_dtype

def python_dtype_to_vivaldi_dtype(python_dtype):
	vivaldi_dtype = ''
	if python_dtype == None: return None
	if python_dtype in [numpy.int8, numpy.uint8]: vivaldi_dtype += 'uchar'
	if python_dtype in [numpy.int16, numpy.uint16]: vivaldi_dtype += 'short'
	if python_dtype in [numpy.int32, numpy.uint32]: vivaldi_dtype += 'int'
	if python_dtype in [numpy.float32, float]: vivaldi_dtype += 'float'
	if python_dtype in [numpy.float64]: vivaldi_dtype += 'double'
	if type(python_dtype) == str: vivaldi_dtype = python_dtype
	assert(vivaldi_dtype!='')
	return vivaldi_dtype


def make_type_name(dtype, shape, last):
	type_name = ''

	if dtype in [numpy.int8, numpy.uint8]: type_name += 'uchar'
	if dtype in [numpy.int16, numpy.uint16]: type_name += 'short'
	if dtype in [numpy.int32, numpy.uint32]: type_name += 'int'
	if dtype in [numpy.float32, float]: type_name += 'float'
	if dtype in [numpy.float64]: type_name += 'double'

	
	if last in [2,3,4]:
		chan = [last]
		type_name += str(last)
		shape.pop()
	else:
		chan = [1]

	return type_name, shape


# vivaldi reader
####################################################################################################################3

def get_buffer_start(buffer_range):
	buffer_start = None
	n = len(buffer_range)
	if n == 1:
		buffer_start = (buffer_range['x'][0])
	elif n == 2:
		buffer_start = (buffer_range['y'][0], buffer_range['x'][0])
	elif n == 3:
		buffer_start = (buffer_range['z'][0], buffer_range['y'][0], buffer_range['x'][0])
	elif n == 4:
		buffer_start = (buffer_range['w'][0], buffer_range['z'][0], buffer_range['y'][0], buffer_range['x'][0])
	if buffer_start == None:
		assert(False)
	return buffer_start

def get_buffer_end(buffer_range):
	buffer_end = None
	n = len(buffer_range)
	if n == 1:
		buffer_end = (buffer_range['x'][1])
	elif n == 2:
		buffer_end = (buffer_range['y'][1], buffer_range['x'][1])
	elif n == 3:
		buffer_end = (buffer_range['z'][1], buffer_range['y'][1], buffer_range['x'][1])
	elif n == 4:
		buffer_end = (buffer_range['w'][1], buffer_range['z'][1], buffer_range['y'][1], buffer_range['x'][1])
	if buffer_end == None:
		assert(False)
	return buffer_end

def get_empty_block(block_size, dim, ms, dtype):
	temp = []
	for elem in range(dim):
		temp.append(block_size)

	if ms != [1]:
		temp += ms

	block = numpy.empty(tuple(temp), dtype=dtype)
	return block

def copy_to_block(data, buffer_start, buffer_end, block, block_position, block_size, dim):
#	block_position and buffer_start is different
#	so we need something mathematics

	s = [0]*dim
	e = [0]*dim
	cmd1 = ''
	cmd2 = ''
	bytes = 0
	for i in range(dim):
		# decide copy range
		if buffer_start[i] <= block_position[i]:
			s[i] = block_position[i]
		elif block_position[i] < buffer_start[i]:
			s[i] = buffer_start[i]
		else:
			assert(False)

		if block_position[i]+block_size <= buffer_end[i]:
			e[i] = block_position[i]+block_size
		elif buffer_end[i] < block_position[i]+block_size:
			e[i] = buffer_end[i]
		else:
			assert(False)

		cmd1 += "%d:%d,"%(s[i]-block_position[i],e[i]-block_position[i])
		cmd2 += "%d:%d,"%(s[i]-buffer_start[i],e[i]-buffer_start[i])
		if bytes == 0:
			bytes = e[i]-s[i]
		else:
			bytes *= e[i]-s[i]

	cmd = 'block['+cmd1[:-1]+'] = data['+cmd2[:-1]+']'
	exec cmd
	return block, bytes

def copy_to_buffer(data, buffer_start, buffer_end, block, block_position, block_size, dim):
#	block_position and buffer_start is different
#	so we need something mathematics

	s = [0]*dim
	e = [0]*dim
	cmd1 = ''
	cmd2 = ''
	bytes = 0
	for i in range(dim):
		# decide copy range
		if buffer_start[i] <= block_position[i]:
			s[i] = block_position[i]
		elif block_position[i] < buffer_start[i]:
			s[i] = buffer_start[i]
		else:
			assert(False)

		if block_position[i]+block_size <= buffer_end[i]:
			e[i] = block_position[i]+block_size
		elif buffer_end[i] < block_position[i]+block_size:
			e[i] = buffer_end[i]
		else:
			assert(False)

		cmd1 += "%d:%d,"%(s[i]-block_position[i],e[i]-block_position[i])
		cmd2 += "%d:%d,"%(s[i]-buffer_start[i],e[i]-buffer_start[i])
		if bytes == 0:
			bytes = e[i]-s[i]
		else:
			bytes *= e[i]-s[i]

	cmd = 'data['+cmd2[:-1]+'] = block['+cmd1[:-1]+']'
	exec cmd




def get_block_maximum(block_size, full_buffer_start, full_buffer_end, block_position, dim):
	s = [0]*dim
	e = [0]*dim
	bytes = 0
	for i in range(dim):
		# decide copy range
		if full_buffer_start[i] <= block_position[i]:
			s[i] = block_position[i]
		elif block_position[i] < full_buffer_start[i]:
			s[i] = full_buffer_start[i]
		else:
			assert(False)

		if block_position[i]+block_size <= full_buffer_end[i]:
			e[i] = block_position[i]+block_size
		elif full_buffer_end[i] < block_position[i]+block_size:
			e[i] = full_buffer_end[i]
		else:
			assert(False)
		
		if bytes == 0:
			bytes = e[i]-s[i]
		else:
			bytes *= e[i]-s[i]

	return bytes

def next_block_position(block_position, block_size, adjusted_buffer_start, buffer_end):
	block_position = list(block_position)
	block_position[0] += 8
	i = 0
	n = len(block_position)
	while True:
		if block_position[i] >= buffer_end[i]:
			block_position[i] = adjusted_buffer_start[0]
			if i + 1 == n: return False
			block_position[i+1] += block_size
		else:
			break
		i += 1
	return tuple(block_position)


# etc
################################################################################################

def divide_args(args):
	### <Summary> :: divide arguments and return list of args
	### Type of args :: str
	### Return type :: list
	args_list = []
	t = args
	last = 0
	bcnt = 0
	i = 0
	temp = ''
	#remove blanket
	for s in args:
		if s in ['[','{','(']:bcnt = bcnt + 1
		if s in [']','}',')']:bcnt = bcnt - 1
		if s == ',' and bcnt == 0:
			temp = args[last:i]
			if temp.isdigit(): temp = int(args[last:i])
			args_list.append(temp)
			last = i + 1
		i = i + 1
	if last < i:
		temp = args[last:i]
		if temp.isdigit(): temp = int(args[last:i])
		args_list.append(temp)

	return args_list


def as_python(input):
	aa = ''
	if type(input) == dict:
		aa += '{'
		cnt = 0

		for key in input:
			if cnt > 0:
				aa += ','
			cnt += 1	
			aa += key+":"
			aa += as_python(input[key])
		aa += '}'
			
	elif type(input) in [list, tuple]:
		aa += '('
		cnt = 0
		for elem in input:
			if cnt > 0 :
				aa += ','
			cnt += 1
			aa += as_python(elem)
		aa += ')'
	else:
		aa += str(input)
	return aa


def split_file_name_and_extension(file_name):
	p = file_name.rfind('.')
	extension = file_name[p+1:]
	file_name = file_name[:p]
	return file_name, extension

def write_dat(file_name, buffer_shape, chan, dtype):
	e = os.system("mkdir -p result")
	block_size = 8
	
	f = open('./result/%s.dat'%(file_name),'w')
	f.write("ObjectFileName: %s.raw\n"%(file_name))
	sfw = ''
	for elem in buffer_shape:
		sfw += str(elem) + ' '
		
	f.write('Resolution: %s\n'%(sfw))
	f.write('chan: %s\n'%(str(chan)))
	f.write('Format: %s\n'%(dtype))
	f.close()





def print_bytes(bytes):
	ss = ''
	units = ['','K','M','G','T']
	unit = ''
	bytes = float(bytes)
	for alpha in units:
		unit = alpha
		if bytes > 1024:
			bytes /= 1024
		else:
			break

	return "%.2f%sBytes"%(bytes,unit)




def remove_comment(in_code, flag='#'):
	new_code = ""
	for line in in_code.splitlines():
		# remove comment
		n = len(line)
		i = 0
		qcnt=0
		dqcnt = 0
		while i < n:
			if line[i] == "'": qcnt = (qcnt+1)%2
			if line[i] == '"': dqcnt = (dqcnt+1)%2
			if qcnt == 1 or dqcnt == 1:
				i += 1
				continue
			if line[i] == flag:
				line = line[:i]
				break
			i += 1
		# remove empty line
		if len(line) == 0:continue
		# find every writing occur
		new_code += line + '\n'
		
	if len(new_code) == 0: return new_code
	return new_code[:-1]

def remove_enter(in_code):
	new_code = ""
	
	for line in in_code.splitlines():
		contents = line.strip()
		# remove empty line
		if len(contents) == 0:continue
		# find every writing occur
		if contents.lstrip().startswith('.'):
			new_code = new_code.rstrip() + contents.lstrip() + '\n'
		else:
			new_code += line + '\n'

	if len(new_code) == 0: return new_code				
	return new_code[:-1]
