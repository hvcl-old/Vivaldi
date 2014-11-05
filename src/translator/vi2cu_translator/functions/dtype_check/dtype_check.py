if True:
	# data type list
	# boolean
	# short
	# short2
	# short3
	# short4
	# unsigned short
	# unsigned short2
	# unsigned short3
	# unsigned short4
	# integer
	# integer2
	# integer3
	# integer4
	# unsigned integer
	# unsigned integer2
	# unsigned integer3
	# unsigned integer4
	# float
	# float2
	# float3
	# float4
	# unsigned float
	# unsigned float2
	# unsigned float3
	# unsigned float4

	# operator
	# unary
	# +
	# -
	# !, not
	# ~
	# 
	# binary
	# .
	# *
	# /
	# %
	# +
	# -
	# <
	# <=
	# >
	# >=
	# ==
	# !=
	# &
	# ^
	# |
	# &&, and
	# ||, or
	# =

	#

	# Usual Arithmetic conversions
	#If either operand is of type long double, the other operand is converted to type long double.
	#If the above condition is not met and either operand is of type double, the other operand is converted to type double.
	#If the above two conditions are not met and either operand is of type float, the other operand is converted to type float.
	#If the above three conditions are not met (none of the operands are of floating types), then integral conversions are performed on the operands as follows:
	#If either operand is of type unsigned long, the other operand is converted to type unsigned long.
	#If the above condition is not met and either operand is of type long and the other of type unsigned int, both operands are converted to type unsigned long.
	#If the above two conditions are not met, and either operand is of type long, the other operand is converted to type long.
	#If the above three conditions are not met, and either operand is of type unsigned int, the other operand is converted to type unsigned int.
	#If none of the above conditions are met, both operands are converted to type int.



	# VIVALDI Arithmetic conversions
	#   this conversions is referenced Usual Arithmetic conversions of Microsoft Developer Networks
	#  and addition of some rules from personal experience and rules for CUDA data type


	# if either operand is 4 set, the other operand is converted to 4 set
	# if the above condition is not met and either operand is 3 set, the other operand is converted to 3 set
	# if the above condition is not met and either operand is 2 set, the other operand is converted to 2 set
	#
	# after above rules
	# if either operand is singed, the other operand is converted to signed
	# if either operand is float, the other operand is converted to type float
	# if the above condition is not met and either operand is integer, the other operand is converted to type integer
	# if the above two condition is not met and either operand is short, the other operand is converted to type short
	# if the above three condition is not met and either operand is char, the other operand is converted to type char
	# if the above four condition is not met and either operand is boolean, the other operand is converted to type boolean
	#
	# after above rule apply
	# if either operand is unsigned float, the other operand is converted to unsigned float
	# if the above condition is not met and either operand is unsigned integer, the other operand is converted to unsigned integer
	# if the above two condition is not met and either operand is unsigned short, the other operand is converted to unsigned short
	# if the above three condition is not met and either operand is unsigned char, the other operand is converted to unsigned char
	#

	# remaining is not defined

	# unary operator
	# operator	boolean		=	boolean
	#
	# binary operator
	# unsigned rule
	# unsigned	operator	unsigned	= unsigned
	# unsigned	operator	signed		= signed
	# signed	operator	unsigned	= signed
	# signed	operator	signed		= signed
	#
	#
	# byte size rule
	#   data type will be selected to minimum data type in the set of data type bigger than both operands
	# and something 3 will follows "helper_math.h" in the CUDA SDK
	#
	# boolean	operator	boolean		= boolean
	# boolean	operator	short		= short
	# boolean	operator	integer		= integer
	# boolean	operator	float		= float
	#
	# float		operator	float		= float
	# float		operator	float2		= float2
	# float		operator	float3		= float3
	# float		operator	float4		= float4
	pass


import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')

path = VIVALDI_PATH+'/src/translator'
if path not in sys.path:
	sys.path.append(path)

from common_in_translator import *
from general.divide_line.divide_line import divide_line

def binary_conversion(front_dtype, back_dtype):

	# VIVALDI Arithmetic conversions for binary operator
	result_dtype = ''
	
	# if either operand is 4 set, the other operand is converted to 4 set
	if front_dtype.endswith('4'):	
		if back_dtype.endswith('4'): pass # nothing 
		elif back_dtype.endswith('3'): back_dtype = back_dtype[:-1] + '4'
		elif back_dtype.endswith('2'): back_dtype = back_dtype[:-1] + '4'
		else: back_dtype += '4'
	if back_dtype.endswith('4'):	
		if front_dtype.endswith('4'): pass # nothing 
		elif front_dtype.endswith('3'): front_dtype = front_dtype[:-1] + '4'
		elif front_dtype.endswith('2'): front_dtype = front_dtype[:-1] + '4'
		else: front_dtype += '4'
	
	# if the above condition is not met and either operand is 3 set, the other operand is converted to 3 set
	if front_dtype.endswith('3'):	
		if back_dtype.endswith('3'): pass # nothing 
		elif back_dtype.endswith('2'): back_dtype = back_dtype[:-1] + '3'
		else: back_dtype += '3'
	if back_dtype.endswith('3'):
		if front_dtype.endswith('3'): pass # nothing 
		elif front_dtype.endswith('2'): front_dtype = front_dtype[:-1] + '3'
		else: front_dtype += '3'
	
	# if the above condition is not met and either operand is 2 set, the other operand is converted to 2 set
	if front_dtype.endswith('2'):
		if back_dtype.endswith('2'): pass # nothing 
		else: back_dtype += '2'
	if back_dtype.endswith('2'):
		if front_dtype.endswith('2'): pass # nothing 
		else: front_dtype += '2'
		
	# after above rules
	# if either operand is singed, the other operand is converted to signed
	f_flag = front_dtype.startswith('unsigned')
	b_flag = back_dtype.startswith('unsigned')
	if f_flag and not b_flag: f_flag.replace('unsigned', '')
	elif not f_flag and b_flag: b_flag.replace('unsigned', '')
	
	# if either operand is float, the other operand is converted to type float
	if front_dtype in ['float','float_constant']: back_dtype = 'float'
	elif back_dtype in ['float','float_constant']: front_dtype = 'float'
	
	# if the above condition is not met and either operand is integer, the other operand is converted to type integer
	if front_dtype in ['integer','integer_constant']: back_dtype = 'integer'
	elif back_dtype in ['integer','integer_constant']: front_dtype = 'integer'
	
	# if the above two condition is not met and either operand is short, the other operand is converted to type short
	if front_dtype in ['short']: back_dtype = 'short'
	elif back_dtype in ['short']: front_dtype = 'short'
	
	# if the above three condition is not met and either operand is char, the other operand is converted to type char
	if front_dtype in ['char']: back_dtype = 'char'
	elif back_dtype in ['char']: front_dtype = 'char'
	
	# if the above four condition is not met and either operand is boolean, the other operand is converted to type boolean
	if front_dtype in ['boolean']: back_dtype = 'boolean'
	elif back_dtype in ['boolean']: front_dtype = 'boolean'
	
	# after above rule apply
	# if either operand is unsigned float, the other operand is converted to unsigned float
	if front_dtype in ['unsigned float','unsigned float_constant']: back_dtype = 'unsigned float'
	elif back_dtype in ['unsigned float','unsigned float_constant']: front_dtype = 'unsigned float'
	
	# if the above condition is not met and either operand is unsigned integer, the other operand is converted to unsigned integer
	if front_dtype in ['unsigned integer','unsigned integer_constant']: back_dtype = 'unsigned integer'
	elif back_dtype in ['unsigned integer','unsigned integer_constant']: front_dtype = 'unsigned integer'
	
	# if the above two condition is not met and either operand is unsigned short, the other operand is converted to unsigned short
	if front_dtype in ['unsigned short']: back_dtype = 'unsigned short'
	elif back_dtype in ['unsigned short']: front_dtype = 'unsigned short'
	
	# if the above three condition is not met and either operand is unsigned char, the other operand is converted to unsigned char
	if front_dtype in ['unsigned char']: back_dtype = 'unsigned char'
	elif back_dtype in ['unsigned char']: front_dtype = 'unsigned char'
	
	return front_dtype

def binary_multiplication(front_dtype, back_dtype):
	# data type calculation	
	binary_dtype = binary_conversion(front_dtype, back_dtype)
	
	return binary_dtype
	
def binary_division(front_dtype, back_dtype):
	# data type calculation	
	binary_dtype = binary_conversion(front_dtype, back_dtype)
	
	return binary_dtype

def binary_addition(front_dtype, back_dtype):
	# data type calculation	
	binary_dtype = binary_conversion(front_dtype, back_dtype)
	
	return binary_dtype
	
def binary_minus(front_dtype, back_dtype):
	# data type calculation	
	binary_dtype = binary_conversion(front_dtype, back_dtype)
	
	return binary_dtype

func_dict = {}
func_dict['if'] = ''
func_dict['elif'] = ''
func_dict['else if'] = ''
func_dict['else'] = ''
func_dict['for'] = ''
func_dict['while'] = ''
func_dict['print'] = ''
func_dict['normalize'] = ''
func_dict[''] = ''
func_dict['diffuse'] = 'float3'
func_dict['phong'] = 'float3'
func_dict['laplacian'] = 'float3'
func_dict['pow'] = 'float'
func_dict['sqrt'] = 'float'
func_dict['expf'] = 'float'
func_dict['exp2f'] = 'float'
func_dict['exp10f'] = 'float'

func_dict['lerp'] = ''
func_dict['clamp'] = ''
func_dict['dot'] = ''
func_dict['length'] = 'float'
func_dict['floor'] = ''
func_dict['ceil'] = ''
func_dict['reflect'] = ''
func_dict['fabs'] = ''
func_dict['floor'] = ''
func_dict['ceil'] = ''
func_dict['fminf'] = ''
func_dict['fmaxf'] = ''

func_dict['min']='float'
func_dict['max']='float'


func_dict['make_uchar'] = 'uchar'

# 2chan 
func_dict['make_int2'] = 'int2'
func_dict['int2.x'] = 'int'
func_dict['int2.y'] = 'int'
func_dict['make_float2'] = 'float2'
func_dict['float2.x'] = 'float'
func_dict['float2.y'] = 'float'

# 3chan
func_dict['make_int3'] = 'int3'
func_dict['int3.x'] = 'int'
func_dict['int3.y'] = 'int'
func_dict['int3.z'] = 'int'
func_dict['make_float3'] = 'float3'
func_dict['float3.x'] = 'float'
func_dict['float3.y'] = 'float'
func_dict['float3.z'] = 'float'

# 4chan
func_dict['make_int4'] = 'int4'
func_dict['int4.x'] = 'int'
func_dict['int4.y'] = 'int'
func_dict['int4.z'] = 'int'
func_dict['int4.w'] = 'int'
func_dict['make_float4'] = 'float4'
func_dict['float4.x'] = 'float'
func_dict['float4.y'] = 'float'
func_dict['float4.z'] = 'float'
func_dict['float4.w'] = 'float'

func_dict['line_iter'] = 'line_iter'
func_dict['line_iter.begin'] = 'float3'
func_dict['line_iter.direction'] = 'float3'
func_dict['line_iter.next'] = 'float3'

func_dict['plane_iter'] = 'plane_iter'
func_dict['plane_iter.begin'] = 'float2'
func_dict['plane_iter.next'] = 'float2'

func_dict['cube_iter'] = 'cube_iter'
func_dict['cube_iter.begin'] = 'float3'
func_dict['cube_iter.next'] = 'float3'

func_dict['orthogonal_iter'] = 'line_iter'
func_dict['perspective_iter'] = 'line_iter'

# 1D data query functions
func_dict['point_query_1d'] = ''
func_dict['linear_query_1d'] = ''

# 2D data query functions
func_dict['linear_query_2d'] = 'float1'
func_dict['linear_gradient_2d'] = 'float2'
func_dict['point_query_2d'] = ''
func_dict['cubic_query_2d'] = 'float1'

# 3D data query functions
func_dict['point_query_3d'] = ''

func_dict['linear_query_3d'] = 'float1'
func_dict['linear_gradient_3d'] = 'float3'

func_dict['cubic_query_3d'] = 'float1'
func_dict['cubic_gradient_3d'] = 'float3'

func_dict['get_ray_origin'] = 'float3'
func_dict['get_ray_direction'] = 'float3'
func_dict['get_start'] = 'float3'
func_dict['get_end'] = 'float3'

func_dict['intersectCube'] = 'float2'

func_dict['RGBA'] = 'RGBA'
func_dict['RGB'] = 'RGB'

# viewer function
func_dict['alpha_compositing'] = 'float4'
func_dict['alpha_compositing_wo_alpha'] = 'float4'
func_dict['transfer'] = 'float4'

def get_function_return_dtype(func_name, args_list, dtype_list, local_dict):
	sfn = func_name.strip()
	
	# query functions
	if sfn in query_list:
		f_a = args_list[0]
		dtype = dtype_list[0]
		if dtype.endswith('_volume'):
			dtype = dtype[:-7]

		in_type = dtype_convert_in_query(dtype)
		exception_list = ['linear_gradient_2d']
		
		if dtype_list[0] == 'Unknown':
			dtype = dtype_list[0]
			found_dtype = 'float_volume'
			add_dtype(args_list[0], found_dtype, local_dict)
			add_dtype(args_list[0]+'_DATA_RANGE', 'VIVALDI_DATA_RANGE', local_dict)	
			dtype_list[0] = found_dtype
			
			
		if sfn in exception_list:
			return func_dict[sfn]
		return in_type

	# common math functions
	if sfn in ['normalize']:
		return dtype_list[0]

	# common functions
	if sfn in func_dict:
		return func_dict[sfn]

	print "VIVALDI Warning"
	print "-----------------------"
	print "Vivaldi compiler don't know the function"
	print "Function name: ", func_name
	print "-----------------------"
	return 'Unknown'

def Function_call(elem_list, dtype_list, local_dict, dtype_dict):

	# preprocessing
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	
	# implementation
	while i < n:
		flag = True
		elem = elem_list[i]
		if dtype_list[i] is not 'operator' and i+1 < n and elem_list[i+1][0] == '(' and elem_list[i+1][-1] == ')':
			# function_name + Tuple
			new_elem = elem_list[i] + elem_list[i+1]
			
			args = elem_list[i+1]
			arg_list = divide_line(args[1:-1])
			
			arg_dtype_list = []	
			for arg in arg_list:
				dtype = find_dtype(arg, local_dict)
				arg_dtype_list.append(dtype)
			
			func_name = elem_list[i]
			j = i
			while j > 0:
				if elem_list[j-1] == '.':
					func_name = local_dict[ elem_list[j-2] ] + '.' + func_name
					j -= 2
				else: break
				
			
			if func_name in dtype_dict:
				dtype_dict[func_name] = 'function'
			
			dtype = get_function_return_dtype(func_name, arg_list, arg_dtype_list, local_dict)
			
			new_dtype_list.append(dtype)
			new_elem_list.append(new_elem)
			i += 2
			flag = False
	
		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
	
	return new_elem_list, new_dtype_list
	
def Array_subscripting(elem_list, dtype_list):
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	while i < n:
		flag = True
		elem = elem_list[i]
		#if dtype_list[i].endswith('_volume') and i+1 < n and elem_list[i+1][0] == '[' and elem_list[i+1][-1] == ']':
		if dtype_list[i] is not 'operator':
			if i+1 < n and elem_list[i+1][0] == '[' and elem_list[i+1][-1] == ']':
			
				new_elem = elem_list[i]
				dtype = dtype_list[i]
				if dtype.endswith('_volume'):
					dtype = dtype.replace('_volume','')
				
				while i+1 < n and elem_list[i+1][0] == '[' and elem_list[i+1][-1] == ']':
					# volume + []
					new_elem += elem_list[i+1]	
					i += 1
				# not formal
				new_dtype_list.append(dtype)
				new_elem_list.append(new_elem)
				flag = False
	
		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
		
	return new_elem_list, new_dtype_list

def get_dot_dtype(before, before_dtype, after, after_dtype):
	# dot is element selection by reference
	# after dot can be function or variable both

	# element + '.' + element = Function or variable
	if before_dtype == 'integer_constant' and after_dtype == 'integer_constant':
		return 'float_constant'
	
	
	key = before_dtype + '.' + after
	if key in func_dict:
		return func_dict[key]
	
#	print "BEFORE"
#	print before, before_dtype
#	print after, after_dtype
	
	return 'Unknown'
	
def Second_level(elem_list, dtype_list, local_dict, dtype_dict):
	# initialization
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	# implementation
	#######################################
	while i < n:
		flag = True
		
		if (dtype_list[i] is not 'operator' 
			and i+1 < n and elem_list[i+1][0] == '(' 
			and elem_list[i+1][-1] == ')'): # Function call check
			# function_name + Tuple
			new_elem = elem_list[i] + elem_list[i+1]
			
			args = elem_list[i+1]
			find_dtype(args[1:-1], local_dict)
			
			func_name = elem_list[i]
			arg_list = divide_line(args[1:-1])
			arg_dtype_list = []	
			for arg in arg_list:
				if arg in local_dict:
					dtype = local_dict[arg]
				else:
					dtype = 'Unknown'
				arg_dtype_list.append(dtype)
			dtype = get_function_return_dtype(func_name, arg_list, arg_dtype_list, local_dict)
			
			new_dtype_list.append(dtype)
			new_elem_list.append(new_elem)
			i += 2
			flag = False
		elif (elem_list[i][0] == '(' 
			and elem_list[i][-1] == ')'):
			# (elem) case
			new_elem = elem_list[i]
			args = elem_list[i]
			dtype = find_dtype(args[1:-1], local_dict)
			new_dtype_list.append(dtype)
			new_elem_list.append(new_elem)
			i += 1
			flag = False
		elif False:
			# Array check
			pass
		elif i+2 < n and elem_list[i+1] == '.':
			# dot check
			# : Element selection by reference
			
			before = elem_list[i]
			after = elem_list[i+2]
			
			before_dtype = dtype_list[i]
			after_dtype = dtype_list[i+2]
			
			new_elem = before+elem_list[i+1]+after
			new_elem_list.append(new_elem)
			dtype = get_dot_dtype(before, before_dtype, after, after_dtype)
			new_dtype_list.append(dtype)
			dtype_dict[new_elem] = dtype
			i+=2
			flag = False
				
		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1	

	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	return new_elem_list, new_dtype_list
	
def Third_level(elem_list, dtype_list, local_dict, dtype_dict):
	# initialization
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	# implementation
	#######################################
	while i < n:
		flag = True
		if elem_list[i] in ['+','-']:
			if i-1 >= 0 and is_operator(elem_list[i-1]):
				new_elem = elem_list[i]+elem_list[i+1]
				new_elem_list.append(new_elem)
				
				dtype = dtype_list[i+1]
				new_dtype_list.append(dtype)
				dtype_dict[new_elem] = dtype
				i += 1
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
	
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	return new_elem_list, new_dtype_list
	
def Fifth_level(elem_list, dtype_list, local_dict, dtype_dict):
	# initialization
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	# implementation
	#######################################
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['*','/']:
			if not is_operator(elem_list[i]) and not is_operator(elem_list[i+2]):
				new_elem = elem_list[i]+elem_list[i+1] +elem_list[i+2]
				new_elem_list.append(new_elem)
				
				front_dtype = dtype_list[i]
				back_dtype = dtype_list[i+2]
				dtype = 'Unknown'
				if elem_list[i+1] == '*':
					dtype = binary_multiplication(front_dtype, back_dtype)
				elif elem_list[i+1] == '/':
					dtype = binary_division(front_dtype, back_dtype)
					
				new_dtype_list.append(dtype)
				dtype_dict[new_elem] = dtype
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
		
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	return new_elem_list, new_dtype_list
	
def Sixth_level(elem_list, dtype_list, local_dict, dtype_dict):
	# initialization
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	# implementation
	#######################################
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['+','-']:
			if not is_operator(elem_list[i]) and not is_operator(elem_list[i+2]):
				new_elem = elem_list[i]+elem_list[i+1] +elem_list[i+2]
				new_elem_list.append(new_elem)
				
				front_dtype = dtype_list[i]
				back_dtype = dtype_list[i+2]
				if elem_list[i+1] == '+':
					dtype = binary_addition(front_dtype, back_dtype)
				elif elem_list[i+1] == '-':
					dtype = binary_minus(front_dtype, back_dtype) # depend on addition and subtraction rule
				new_dtype_list.append(dtype)
				dtype_dict[new_elem] = dtype
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
		
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	return new_elem_list, new_dtype_list
		
def dtype_check(elem_list, dtype_list, local_dict):
	# return variable
	dtype_dict = {}
	
	# find integer
	i = 0
	n = len(elem_list)
	new_dtype_list = []
	
	while i < n:
		flag = True
		elem = elem_list[i]
		
		if elem.isdigit():
			dtype = 'integer_constant'
			new_dtype_list.append(dtype)
			dtype_dict[elem] = dtype
			flag = False
			
		if flag:
			new_dtype_list.append(dtype_list[i])
		
		i += 1
	dtype_list = new_dtype_list
	
	# second level
	# .
	# Function Call ()
	# Array subscripting
	elem_list, dtype_list = Second_level(elem_list, dtype_list, local_dict, dtype_dict)

	# third level
	# unary + and -	
	# logical not
	elem_list, dtype_list = Third_level(elem_list, dtype_list, local_dict, dtype_dict)
	
	# fifth level
	# multiplication 
	# division
	elem_list, dtype_list = Fifth_level(elem_list, dtype_list, local_dict, dtype_dict)
	
	# sixth level
	# addition
	# subtraction
	elem_list, dtype_list = Sixth_level(elem_list, dtype_list, local_dict, dtype_dict)
	
	# eighth level
	# Less than
	# Less than or equal to
	# Greater than
	# Greater than or equal to
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['>','>=','<','<=']:
			if not is_operator(elem_list[i]) and not is_operator(elem_list[i+2]):
				new_elem = elem_list[i]+elem_list[i+1] +elem_list[i+2]
				new_elem_list.append(new_elem)
				
				dtype = 'bool'
				new_dtype_list.append(dtype)
				dtype_dict[new_elem] = dtype
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	"""
	# ninth level
	# Equal to
	# Not equal to
	i = 0
	new_elem_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['==', '!=']:
			if elem_list[i] is not operator and elem_list[i+2] is not operator:
				new_elem_list.append(elem_list[i]+elem_list[i+1] +elem_list[i+1])
				dtype = 'bool'
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
		i += 1

	# 13th level
	# Logical AND
	# Logical OR
	i = 0
	new_elem_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['and', 'or']:
			if elem_list[i] is not operator and elem_list[i+2] is not operator:
				new_elem_list.append(elem_list[i]+elem_list[i+1] +elem_list[i+1])
				dtype = 'bool'
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
		i += 1
	"""
	# 14th
	# Direct assignment
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['=']:
			
			if dtype_list[i] is not 'operator' and elem_list[i+2] is not 'operator':
				new_elem = elem_list[i]+elem_list[i+1]+elem_list[i+2]
				new_elem_list.append(new_elem)
				
				# remove constant mark
				dtype = dtype_list[i+2]
				if dtype.endswith('_constant'):
					dtype = dtype[:-len('_constant')]
				new_dtype_list.append(dtype)
				
				# write output value
				dtype_dict[elem_list[i]] = new_dtype_list[i]
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
	
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	if False:
		# Debug
		print elem_list
		print dtype_list
	
	"""
	# +=
	# -=
	i = 0
	i = 0
	new_elem_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['+=','-=']:
			if elem_list[i] is not operator and elem_list[i+2] is not operator:
				new_elem_list.append(elem_list[i]+elem_list[i+1] +elem_list[i+1])
				dtype = # depend on addition and subtraction rule
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
		i += 1
	
	# *=
	# /=
	i = 0
	new_elem_list = []
	while i < n:
		flag = True
		if i+2 < n and elem_list[i+1] in ['*=','/=']:
			if elem_list[i] is not operator and elem_list[i+2] is not operator:
				new_elem_list.append(elem_list[i]+elem_list[i+1] +elem_list[i+1])
				dtype = # depend on multiplication and division rule
				i += 2
				flag = False

		if flag:
			new_elem_list.append(elem_list[i])
		i += 1

	# 18th
	# , comma. where is it used?
	
	i = 0
	new_elem_list = []
	while i < n:
		elem = elem_list[i]
		i += 1
	
	
	dtype = dtype_list[0]
	return dtype
	"""
	
	# remove if, elif, else, while, for, return, print
	# for debugging purpose
	
	g_list = ['if', 'elif', 'else', 'while', 'for', 'return', 'print']
	i = 0
	n = len(elem_list)
	new_elem_list = []
	new_dtype_list = []
	while i < n:
		flag = True
		if elem_list[i] in g_list:
			new_elem = ''
			while i < n:
				w = elem_list[i]
				new_elem += w
				i += 1
				if w == ':':
					break
			new_elem_list.append(new_elem)
			
			dtype = 'built_in grammar'
			new_dtype_list.append(dtype)
			dtype_dict[new_elem] = dtype
			flag = False
		if flag:
			new_elem_list.append(elem_list[i])
			new_dtype_list.append(dtype_list[i])
		i += 1
	
	elem_list = new_elem_list
	dtype_list = new_dtype_list
	
	return dtype_dict, dtype_list
	
def get_parenthesis(elem):
    # elem is string
    # it checks this is a function or not

    # find parenthesis open
    p_open = ['(','{','[']
	# check from start of string
    for w in elem:
		for parenthesis in p_open:
			if w == parenthesis:
				# this is tuple of array , it's not start from '('
				return w

    return ''

def find_dtype(line, local_dict):
	# Element list from line
	#
	#####################################################################################
	elem_list = divide_line(line)
	# Element type check
	#
	#####################################################################################
	dtype_list = []
	for elem in elem_list:
		# check is recursive
		if is_recursive(elem):
			# check which parenthesis is used
			parenthesis = get_parenthesis(elem)
			
			# if elem is list, than this will find out element data type
			func_name, args = split_elem(elem)
			arg_list = divide_line(args)
			
			if func_name == '':
				# type of element 
				dtype = ''
				if parenthesis == '[': dtype = 'list'
				elif parenthesis == '{':dtype = 'dictionary'
				elif parenthesis == '(': dtype = 'tuple'
				dtype_list.append(dtype)			
		elif is_operator(elem):
			dtype = 'operator'
			dtype_list.append(dtype)
		else:
			dtype = get_dtype(elem, local_dict)
			dtype_list.append(dtype)
	
	# Data type check
	#
	#####################################################################################

	dtype_dict, dtype_list = dtype_check(elem_list, dtype_list, local_dict)
	
	# Update dictionary
	#####################################################################################
	for elem in dtype_dict:
		add_dtype(elem, dtype_dict[elem], local_dict)
	
	if len(dtype_list) > 0:
		return dtype_list[0]

def test(test_input, test_output):

	result = dtype_check(elem_list, local_dict)
