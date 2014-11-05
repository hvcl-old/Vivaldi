import sys, os
VIVALDI_PATH = os.environ.get('vivaldi_path')
VIVALDI_PATH = '/home/hschoi/Vivaldi'
path = VIVALDI_PATH + "/src"
if path not in sys.path:
	sys.path.append(path)
	
from common import *
OPERATORS = ["==",'+=','-=','*=','/=','=','+','-','*','/',"<=","=>","<",">",".",':',',']
WORD_OPERATORS = ['and','or','not','is','if','for','while','in','return','def', 'print', 'from', 'import', 'as']
AXIS = ['x','y','z','w']
	
query_list = ['point_query_1d', 'linear_query_1d', 'laplacian', 
              'point_query_2d', 'linear_query_2d', 'cubic_query_2d',
	          'point_query_3d','linear_query_3d','cubic_query_3d']
gradient_list = ['linear_gradient_2d','linear_gradient_3d','cubic_gradient_3d']
	
def test_cuda_compile(code):
	#attachment = self.attachment

	main = """
			int main()
			{
					return EXIT_SUCCESS;
			}
			"""

	test_code = attachment + '\n' + code + '\n' + main
	
	f = open('tmp_compile.cu','w')
	f.write(test_code)
	f.close()

	e = os.system('nvcc tmp_compile.cu 2>> compile_log')
	if e != 0:
		return False
	return True
	
def skip_string(line, i):
	# this function skip string starts with quotation or double quotation

	# initialization
	###################################################
	n = len(line)
	s_pair = ["'", '"']
	# implementation
	###################################################
	
	w = line[i]
	if w in s_pair:
		i += 1
		quot = w
		while i < n:
			w = line[i]
			if w == quot:
				return True, i+1
			i += 1
	
	return False, i
	
def skip_parenthesis(line, i, inverse=False):
	# this function skip parenthesis 
	
	# initialization
	###################################################
	n = len(line)
	
	pair_start_list = ['[', '{', '(']
	pair_end_list = [']', '}', ')']
	pair_mapping = {'[':']','{':'}','(':')',
					']':'[','}':'{',')':'('}
					
	cnt = 0
	# implementation
	###################################################
	
	if i >= n: 
		return False, i
	if inverse == False:
		w = line[i]
		if w in pair_start_list:
			pair_end = pair_mapping[w]
			pair_start = w
			while i < n:
				w = line[i]
				# skip string
				flag, i = skip_string(line, i)
				if flag:
					continue
				# check this is pair end
				if w == pair_start:
					cnt += 1
				if w == pair_end:
					cnt -= 1
					if cnt == 0:
						return True, i+1
				i += 1
	else:
		w = line[i]
		if w in pair_end_list:
			i -= 1
			pair_start = pair_mapping[w]
			pair_end = w
			while i > 0:
				w = line[i]
				# skip string
				flag, i = skip_string(line, i)
				if flag:
					continue
				# check this is pair starts
				if w == pair_start:
					return True, i - 1
				i -= 1
			
	return False, i

	
# check this element can be divided to smaller elements
def is_recursive(elem):
	# if there are parenthesis,
	# that mean it has arguments
	p_type_list = ['(','[','{']
	
	for p_type in p_type_list:
		if p_type in elem:
			return True

	return False
	
operators = ['.',',','+','-','!','~','*','&','*','/','%','<<','>>','<','<=','>','>=','==','!=','^','|','&&','||','=','+=','-=','*=','/=','%=','<<=','>>=','&=','^=','|=',':']
	
# check this element is operator or not
def is_operator(elem):
	s_elem = elem.strip()		
	if s_elem in operators:
		return True
		
	if s_elem in OPERATORS:
		return True
		
	if s_elem in WORD_OPERATORS:
		return True
	return False
	
# check this is function
def is_function(elem):
	# elem is string
	# it checks this is a function or not

	i = 0
	idx = 0

	# find parenthesis open
	p_open = ['(','{','[']
	# check from start of string
	for w in elem:
		if w in p_open:
			if w != '(':
				# this is tuple of array , because not start with '('
				idx = i
				return False
			else:
				idx = i
				break
		i += 1
		
	# check function name is correct
	name = elem[:idx]
	
	# empty is not function name
	if name == '':
		return False
				
	return True
	
def get_line(code):
	output = ''
	i = 0
	n = len(code)
	while i < n:
		w = code[i]
		if w == '\n': break
		output += w
		i += 1

	return output

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
	elif type(input) in [tuple]:
		aa += '('
		cnt = 0
		for elem in input:
			if cnt > 0 :
				aa += ','
			cnt += 1
			aa += as_python(elem)
		aa += ')'

	elif type(input) in [list]:
		aa += '['
		cnt = 0
		for elem in input:
			if cnt > 0 :
				aa += ','
			cnt += 1
			aa += as_python(elem)
		aa += ']'
	else:
		aa += str(input.strip())
	return aa

def as_list(line):
	from general.divide_line.divide_line import divide_line
	
	elem_list = divide_line(line)
	output = []
	cnt = 0
	temp = ''
	i = 0
	n = len(elem_list)
	while i < n:
		w = elem_list[i]
		if w == ',':
			output.append(temp)
			temp = ''
		else:
			temp += w
			
		i += 1
	
	if temp != '':
		output.append(temp)
	
	return output
		
# find difference of two string
def dif(a, b):

	# a is user result
	# b is correct result
	n = len(a)
	m = len(b)
	if n > m: n = m
	i = 0
	
	while i < n:
		if a[i] != b[i]:

			C = get_line(b[i:])
			R = get_line(a[i:])

			print "DIF two from string", a[i:i+100]
			print "CORRECT"
			print get_line(b[i:])

			print "RESULT"
			print get_line(a[i:])
			return False
		i += 1
	
	if len(a) != len(b):
		print "DIF LENGTH"
		print "CORRET", len(b)
		print "RESULT", len(a)
		return False
	return True
	
# seems like not necessary to be common
################################################################################
def get_dtype(elem, local_dict, special=None):

	if special == None:
		s_elem = elem.strip()
		if s_elem in local_dict:
			return local_dict[s_elem]
		return 'Unknown'
	elif special == 'for':
		s_elem = elem.strip()
		if s_elem in local_dict:
			dtype = local_dict[elem]
			if dtype == 'line_iter': return 'float3'
			if dtype == 'plane_iter': return 'float2'
			if dtype == 'cube_iter': return 'float3'
			
			return dtype
			
		return 'Unknown'
	
def split_elem(elem):
	# split function name and arguments
	p_type_list = [ ('(',')'), ('[',']')] 
	name = ''
	args = ''
	
	for p_type in p_type_list:
		p_st = p_type[0]
		p_end = p_type[1]
		
		# find start idx
		s_idx = elem.find(p_st)
		if s_idx == -1: 
			continue
		
		# check ????
		if 0 <= s_idx:
			# name is until parenthesis start
			name = elem[:s_idx]
			
			# end index need iteration
			# because we have to check, number of parenthesis is correct
			t_idx = s_idx
			while True:
				e_idx = elem.find(p_end, t_idx+1)
				
				if elem[s_idx:e_idx+1].count('(') == elem[s_idx:e_idx+1].count(')'):
					break
				t_idx = e_idx+1
				
			args = elem[s_idx+1:e_idx]
			
			return name, args
			
	return name, args
	
def add_dtype(var_name, dtype, local_dict):
	from general.divide_line.divide_line import divide_line
	# this function add mapping information between variable name and data type
	# add only correct mapping in the local dictionary
	# because this dictionary used for add data type declaration
	
	# there are some incorrect mapping
	# 1. operator is not variable
	# 2. function is not variable
	
	# remove incorrect mapping
	
	# operator is not variable
	if is_operator(var_name): return
	
	# function is not variable
	if is_function(var_name): return
	if dtype == 'function':return
	
	# constant is not variable
	if dtype.endswith('constant'): return
	
	# built in grammar is not variable
	if dtype == 'built_in grammar': return
	
	# Unknown is ?
	if dtype == 'Unknown': 
		#print "???", var_name, dtype
		#return
		pass
	
	# complex is not single variable
	elem_list = divide_line(var_name)
	for elem in elem_list:
		if elem in operators:
			return
			
	# start with parenthesis
	if var_name[0] in ['(','{','[']: return
	
	# data type override
	if var_name not in local_dict: # first add
		local_dict[var_name] = dtype
	else:
		previous = local_dict[var_name]
		
		if previous == 'Unknown': # overwrite Unknown 
			local_dict[var_name] = dtype
		else:
			# check we have to overwrite or not
			#local_dict[var_name] = dtype
			pass
			
	if var_name == 'color' and local_dict['color'] == 'Unknown': # debug
	#	print "BBB", var_name, dtype
	#	print local_dict
		assert(False)
		
def dtype_convert_in_query(in_type):
	temp = 'float'
	if in_type in ['uchar2','float2','short2','int2','double2']: temp = 'float2'
	if in_type in ['RGB','uchar3','short3','int3','float3','double3']: temp = 'float3'
	if in_type in ['RGBA','uchar4','short4','int4','float4','double4']: temp = 'float4'
	return temp
	
def find_main(code):
	# initialization
	###################################################
	
	st = 'def main():'
	idx = -1
	i = 0 
	pair_list = [('[',']'), ('{','}'), ('(',')'), ('"','"'), ("'","'")]
	ip_flag = False
	m_flag = False
	n = len(code)
	# implementation
	###################################################
		
	# find main start 
	while i < n:
		w = code[i]
		if ip_flag: # in the parenthesis
			# check this is parenthesis close symbol
			if w == symbol:
				symbol = ''
				ip_flag = False
		
		else: # out of parenthesis
			for pair in pair_list:
				if w == pair[0]:
					symbol = pair[1]
					ip_flag = True
					break
			
			# find main
		
			if code[i:].startswith('def main():'):
				m_flag = True
				break
				
		i += 1
	
	# find main end
	# there are n case main finish
	# ex1) code end
	# def main()
	# 	...
	#
	# ex2) another function 
	# def main():
	# 	...
	# def func():
	#
	# ex3) indent
	# def main():
	# 	...
	# print 
	
	output = ''
	line = ''
	cnt = 1
	while i < n:
		w = code[i]
		
		line += w
		if w == '\n':
			indent = get_indent(line)
			if indent == '' and line.strip().startswith('def'):
				# ex2 and ex3
				if cnt == 0:
					break
				cnt -= 1
			
			output += line
			line = ''
		i += 1
		
	if line != 'w':
		output += line
	return output
	
def dtype_matching(function_call_argument_list, function_argument_list, dtype_dict):
	# matching data type between function arguments and function call augments
	i = 0
	
	n = len(function_call_argument_list)
	m = len(function_argument_list)
	if n != m:
		print "WARNING"
		print "---------------------"
		print "Function argument number is not matching"		
		print function_call_argument_list
		print function_argument_list
		print "---------------------"
	
	new_dtype_dict = {}
	for function_argument in function_argument_list:
		function_call_argument = function_call_argument_list[i]
		
		if function_call_argument in dtype_dict:
			new_dtype_dict[function_argument] = dtype_dict[function_call_argument]
		else:
			if function_call_argument in AXIS:
				new_dtype_dict[function_argument] = 'integer'
			else:
				# 2 case
				# float number or volume
				
				if function_call_argument.isdigit(): # float
					new_dtype_dict[function_argument] = 'float'
				else: # volume
					#new_dtype_dict[function_argument] = 'Unknown_volume'
					new_dtype_dict[function_argument] = 'float_volume' # Unknown is considered float
				
		i += 1
	return new_dtype_dict

def NOTHING():
	pass
