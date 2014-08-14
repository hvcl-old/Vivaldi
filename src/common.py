# this is common variable or functions

# global variables
#####################################################################
import sys, os
VIVALDI_PATH = os.environ.get('vivaldi_path')
path = VIVALDI_PATH + "/src/translator"
if path not in sys.path:
	sys.path.append(path)

attachment = ''

# import 
"""
try:
	from general.divide_line.divide_line import divide_line
except:
	pass
	
try:
	from general.general import *
except:
	pass
"""
	
# functions
#####################################################################

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

make_arg = ['val', 'val.x, val.y', 'val.x, val.y, val.z','val.x, val.y, val.z, val.w']	
def get_writing_1d_functions():
	writing_1d_template = """
	__global__ void writing_1d_%s(%s* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	%s* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		
		int x = x_hschoi + x_start;
		
		if(x_end <= x)return;
		
		int idx = (x-rb_DATA_RANGE->buffer_start.x);
		
		float%s val = point_query_1d<float%s>(A, x, A_DATA_RANGE);
		%s a = %s%s%s(%s);
		rb[idx] = a;
	}
						"""
	type_name = None
	attachment = ""
	i = 0
	for dt in ['char','uchar','short','ushort','int','uint','float','double']:
		i = 0
		for chan in ['1','2','3','4']:
		
			if chan == '1':
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_1d_template%(dt, type_name,
							type_name,
							'', '',
							type_name,
							make_func, dt, chan, make_arg[i]) + '\n'

			else:
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_1d_template%(type_name, type_name,
							type_name,
							chan, chan,
							type_name,
							make_func, dt, chan, make_arg[i]) + '\n'
			
			i += 1
						
	return attachment
	
def get_writing_2d_functions():
	writing_2d_template = """
	__global__ void writing_2d_%s(%s* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	%s* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		
		if(x_end <= x || y_end <= y)return;
		
		int idx =
		(y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float%s val = point_query_2d<float%s>(A, x, y, A_DATA_RANGE);
		%s a = %s%s%s(%s);
		rb[idx] = a;
	}
						"""

	type_name = None
	attachment = ""
	for dt in ['char','uchar','short','ushort','int','uint','float','double']:
		i = 0
		for chan in ['1','2','3','4']:
			if chan == '1':
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_2d_template%(dt, dt,
							dt,
							'', '',
							dt, '', '', '', make_arg[i]) + '\n'

			else:
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_2d_template%(type_name, type_name,
							type_name,
							chan, chan,
							type_name,
							make_func, dt, chan, make_arg[i]) + '\n'
			i += 1
		
	i = 2		
	for dt in ['RGB', 'RGBA']:
		
		attachment += writing_2d_template%(dt, dt,
		dt,
		len(dt), len(dt),
		dt, dt, '', '', make_arg[i]) + '\n'		
		i += 1

	return attachment
	
def get_writing_3d_functions():
	writing_3d_template = """
	__global__ void writing_3d_%s(%s* rb, VIVALDI_DATA_RANGE* rb_DATA_RANGE,
	%s* A, VIVALDI_DATA_RANGE* A_DATA_RANGE,
	int z_start, int z_end, int y_start, int y_end, int x_start, int x_end)
	{
		//parallel variables
		int x_hschoi = threadIdx.x + blockDim.x * blockIdx.x;
		int y_hschoi = threadIdx.y + blockDim.y * blockIdx.y;
		int z_hschoi = threadIdx.z + blockDim.z * blockIdx.z;
		
		int x = x_hschoi + x_start;
		int y = y_hschoi + y_start;
		int z = z_hschoi + z_start;
		
		if(x_end <= x || y_end <= y || z_end <= z)return;
		
		int idx =
		(z-rb_DATA_RANGE->buffer_start.z)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)*(rb_DATA_RANGE->buffer_end.y-rb_DATA_RANGE->buffer_start.y)
		+ (y-rb_DATA_RANGE->buffer_start.y)*(rb_DATA_RANGE->buffer_end.x-rb_DATA_RANGE->buffer_start.x)
		+ (x-rb_DATA_RANGE->buffer_start.x);
		
		float%s val = point_query_3d<float%s>(A, x, y, z, A_DATA_RANGE);
		%s a = %s%s%s(%s);
		rb[idx] = a;
	}
						"""

	type_name = None
	attachment = ""
	for dt in ['char','uchar','short','ushort','int','uint','float','double']:
		i = 0
		for chan in ['1','2','3','4']:
			if chan == '1':
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_3d_template%(dt, type_name,
							type_name,
							'', '',
							type_name,
							make_func, dt, chan, make_arg[i]) + '\n'

			else:
				make_func = 'make_'
				type_name = dt+chan
				attachment += writing_3d_template%(type_name, type_name,
							type_name,
							chan, chan,
							type_name,
							make_func, dt, chan, make_arg[i]) + '\n'
			i += 1

	return attachment

def load_attachment():
	includes = '#include "%s/src/py-src/helper_math.h"'%(VIVALDI_PATH) + '\n'
	attachment = includes + read_file(VIVALDI_PATH+'/src/py-src/Vivaldi_cuda_attachment.cu')
	attachment += 'extern "C"{\n'

	writing_1d_functions = get_writing_1d_functions()
	attachment += writing_1d_functions
	writing_2d_functions = get_writing_2d_functions()
	attachment += writing_2d_functions
	writing_3d_functions = get_writing_3d_functions()
	attachment += writing_3d_functions

	attachment += '}\n'				
	return attachment
	
def find_code(func_name, code):
	# initialization
	###################################################
	st = 'def '+func_name+'('
	output = ''
	s_idx = code.find(st)
	if s_idx == -1:
		print "CAN NOT FIND FUNCTION"
		print "FUNCTION WANT TO FIND"
		print func_name
		print "CODE"
		print code
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
	
# get indent of the input line
def get_indent(line): 
	s_line = line.strip()
	i_idx = line.find(s_line)
	indent = line[:i_idx]

	return indent
	
# initialization
#####################################################################

attachment = load_attachment()
