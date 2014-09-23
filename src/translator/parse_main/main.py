# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
path = VIVALDI_PATH+'/src/translator'
if path not in sys.path:
	sys.path.append(path)

from common_in_translator import *
from general.divide_line.divide_line import divide_line
#####################################################################

import ast

def remove_head(code): # not used
	head = 'def main():'
	idx = code.find('def main():')
	output = code[idx+len(head):]
	return output

def remove_one_indent(code): # not used
	output = code.replace('\n    ','\n')
	return output

def remove_comment(code, flag='#'):
    output = ''
    flag = True
    for w in code:
        if w == '#': flag = False
        if w == '\n': flag = True
        if flag: output += w
    return output	

def is_assignment(line): # not used
	# check assignment occur in this line
	# algorithm, find assignment operator
	
	# assignment example
	
	# True example
	# a = b # OK
	# a.b = c # OK
	# a[b] = c # OK
	# a[b+c] = d # OK
	# a[b+c] = f(x=3,y=4) # OK
	# a.b(x=3, y=5) = c # OK because divide_line result is ['a', '.', 'b(x=3,y=5)']
	# a.b[3][4] = c
	
	# False example
	# return a # OK
	# print 'kkk' # OK
	# print 'k = ' + 4 # OK because divide_line result is ['print', 'k=4 ', '+', '4']
	# Identity() # OK
	# f(x=3, y=4) # OK because divide_line result is ['f(x=3, y=4)']
	
	# not exist example
	# a += 1 # preprocessed to a = a + 1

	# initialization
	elem_list = divide_line(line)
	flag = False
	idx = -1
	"""
	for elem in elem_list:
		# find assignment operator
		if elem == '=':
			# find where is the '=' in the line
			# and 
		
	"""
	return flag, idx
	
def get_return_name(line, idx):
	line = str(line[:idx])
	n = len(line)
	i = n-1
	end = n
	output = ''
	while i > 0:
		w = line[i]
		# skip string

		# skip parenthesis
		flag, i = skip_parenthesis(line, i, inverse=True)
		if flag:
			continue

		if w == '(':
			i += 1
			break

		if w == '=':
			end = i-1

		i -= 1

	return line[i:end].strip()

def get_func_name(line):
 
	# find function name in the line
	
	# initialization
	###################################################
	output = ''
	n = len(line)
	i = 0
	pair_list = [('"','"'), ("'","'")]
	ip_flag = False
	p_cnt = 0
	# implementation
	###################################################
		
	idx = line.find(').')
	
	i = idx
	p_cnt = 1
	# find func_name end
	while i > 0:
		w = line[i]
		
		# skip parenthesis
		flag, i = skip_parenthesis(line, i, inverse=True)
		if flag:
			break
		i -= 1
		
	end = i+1
	
	# find func_name start
	start = 0
	
	while i > 0:
		w = line[i]
		
		if w in [' ', '\n', '=', '(', ',']:
			i += 1
			start = i
			break
			
		i -= 1
	
	func_name = line[start:end]
	
	return func_name.strip(), end+1
	
def get_execid(line, idx):
	# initialization
	###################################################
	output = ''
	n = len(line)
	i = idx
	st = '.execid('
	# implementation
	###################################################
	while i < n:
		
		if line[i:].startswith(st):
			i += len(st)
			
			while i < n:
				w = line[i]
				if w == ')':
					break
				output += w
				i += 1
			break
		
		i += 1
		
	if output == '':
		output = '[]'
	return output
	
def get_range(line, idx):
	# parse range in the VIVALDI modifier 
	# there are several way to specify range
	# 1. (volume, x=0:33, y=0:99, z= ....) # integer and range style
	# 2. (variable) # VIVALDI object
	# what we have to care full is 
	# there are output halo definition in the same input
	
	
	# initialization
	###################################################
	# implementation
	###################################################
	
	# there are three type of output range
	
	r_idx = line.find('.range(')
	
	if r_idx == -1:
		return False, False
	else:
		# find args end
		line = str(line[r_idx+len('.range('):])
		n = len(line)
		i = 0
		while i < n:
			w = line[i]
			# skip string
			flag, i = skip_string(line, i)
			if flag: 
				continue
			
			# skip parenthesis
			flag, i = skip_parenthesis(line, i)
			if flag:
				continue
				
			# find assignment
			if w == ')':
				break
			i += 1
		args = line[:i]
		
		if ':' not in args: #???????
			return args, True
			
		# parse argument
		# only x = 0: 3, y = 0: 3
		elem_list = divide_line(args)
		
		output = {}
				
		i = 0
		n = len(elem_list)
		flag = 0
		before = ''
		after = ''
		while i < n:
			w = elem_list[i]
			if w in AXIS:
				if flag == 2:
					output[axis] = (before, after[:-1])
				axis = "'"+w+"'"
				before = ''
				after = ''
				flag = 1
				i += 2
				continue
				
			if w == ':':
				flag = 2
				i += 1
				continue
			
			if flag == 1:
				before += w
			if flag == 2:
				after += w
				
			i += 1
				
		if flag == 2:
			output[axis] = (before, after)
			
		return output, True
	
	return False, False
	
def get_output_halo(line, idx):
		# parse output halo in the VIVALDI modifier 
	# output halo always start from 'halo= ...'
	# what we have to care full is 
	# there are output halo definition in the same input

	# initialization
	###################################################
	# implementation
	###################################################
	
	# there are three type of output range
	
	r_idx = line.find('.output_halo(')
	
	if r_idx == -1:
		return False, False
	else:
		# find args end
		line = str(line[r_idx+ len('.output_halo('):])
		n = len(line)
		i = 0
		while i < n:
			w = line[i]
			# skip string
			flag, i = skip_string(line, i)
			if flag: 
				continue
			
			# skip parenthesis
			flag, i = skip_parenthesis(line, i)
			if flag:
				continue
				
			# find assignment
			if w == ')':
				break
			i += 1
		
		args = line[:i]
		return args, True
	
	return False, False
	
def get_split(line, idx):
	# initialization
	###################################################
	# implementation
	###################################################
	idx = line.find('.split(')
	
	if idx == -1:
		return False, False
	else:
		output = {}
		idx = -1
		while True:
			idx = line.find('.split(', idx+1)
			
			if idx == -1:
				break
			
			# find args end
			arg_line = str(line[idx+len('.split('):])
			n = len(arg_line)
			i = 0
			while i < n:
				w = arg_line[i]
				# skip string
				flag, i = skip_string(arg_line, i)
				if flag: 
					continue
				
				# skip parenthesis
				flag, i = skip_parenthesis(arg_line, i)
				if flag:
					continue
					
				# find assignment
				if w == ')':
					break
				i += 1
			args = arg_line[:i]
			idx += 1
			
			elem_list = divide_line(args)
			
			i = 0
			n = len(elem_list)
			flag = 0
			value = ''
			axis = ''

			var_name = "'"+elem_list[0]+"'"
			output[var_name] = {}
			while i < n:
				w = elem_list[i]
				
				if w in AXIS:
					if flag == 2:
						output[var_name][axis] = value
						
					axis = "'"+w+"'"
					value = ''
					flag = 2
					i += 2
					continue
					
				if flag == 2:
					value += w

				i += 1
				
			if flag == 2:
				output[var_name][axis] = value
				value = ''
				
		return output, True
		
	return False, False
		
def get_args(line):
 
	i = 0
	n = len(line)
	while i < n:
		w = line[i]
		# skip string
		flag, i = skip_string(line, i)
		if flag: 
			continue
		
		# skip parenthesis
		flag, i = skip_parenthesis(line, i)
		if flag:
			continue
			
		# find assignment
		if w == ')':
			break
		i += 1
	args = line[:i]
	
	return args
		
def get_merge(line, idx):

	# ???
	# 
	# merge modifier define which function will be used to merge and order
	

	# initialization
	###################################################
	# implementation
	###################################################
	idx = line.find('.merge(')
	
	if idx == -1:
		return False, False, False
	else:
		# find args end
		line = str(line[idx+len('.merge('):])
		args = get_args(line)
		
		elem_list = divide_line(args)
		
		n = len(elem_list)
		
		# there are two case
		if n == 1: # 1. only function name, front and back order doesn't care

			merge_func = args		
			if merge_func[0] not in ["'",'"']:
				if merge_func[0] == '"': pass
				elif merge_func[0] == "'": pass
				else: merge_func = "'" + merge_func + "'"
			
			merge_order = "'front-to-back'"
			return merge_func, merge_order, True
				
		elif n == 3: # 2. function name and order, order is important
			# n is 3, because comma is included in elem_list
			idx = args.find(',')
		
			merge_func = args[:idx]
			
			if merge_func[0] not in ["'",'"']:
				if merge_func[0] == '"': pass
				elif merge_func[0] == "'": pass
				else: merge_func = "'" + merge_func + "'"
			merge_order = args[idx+1:].strip()
		
			return merge_func, merge_order, True
		
	return False, False, False

def get_halo(line, idx):
	# initialization
	###################################################
	 
	# implementation
	###################################################
	idx = line.find('.halo(')
	
	if idx == -1:
		return False, False
	else:
		output = {}
		idx = -1
		while True:
			idx = line.find('.halo(', idx+1)
			
			if idx == -1:
				break
				
			# find args end
			line = str(line[idx+len('.halo('):])
			args = get_args(line)
			idx += 1
			
			elem_list = divide_line(args)
			c_idx = args.find(',')
			
			before = "'" + args[:c_idx] + "'"
			after = args[c_idx+1:]
			
			output[before] = after
		
		return output, True
		
	return False, False

def get_dtype(line, idx):
	# initialization
	###################################################
	 
	# implementation
	###################################################
	idx = line.find('.dtype(')

	if idx == -1:
		return False, False
	else:
		output = {}
		idx = -1
		while True:
			idx = line.find('.dtype(', idx+1)
			
			if idx == -1:
				break
			
			# find args end
			c_line = str(line[idx+len('.dtype('):])
			args = get_args(c_line)
			idx += 1
			
			elem_list = divide_line(args)
			c_idx = args.find(',')
			
			before = "'" + args[:c_idx] + "'"
			after = "'" + args[c_idx+1:].strip() + '_volume' + "'"
			
			output[before] = after

		return output, True
	
	return False, False

def get_left(line, idx):
 
	line = str(line[idx-1:])
	i = 0
	n = len(line)
	while i < n:
		w = line[i]
		# skip string
		flag, i = skip_string(line, i)
		if flag: 
			continue
		# skip parenthesis
		flag, i = skip_parenthesis(line, i)
		if flag:
			continue
				
		# find assignment
		if w in [',','\n']:
			break
		i += 1

	output = line[i:]
	return output
	
def change_modifier(code):
 
	# this function change modifier to python style
	# because original python don't have grammar like modifier
	
	# initialization
	###################################################
	modifier_list = ['execid','split','range','merge','halo','output_halo','dtype','modifier']
	n  = len(code)
	i = 0
	output = ''
	line = ''
	m_flag = False
	m_idx = -1
	# implementation
	###################################################
	def change_line(line):
		# change to below form
		# 'run_function( func_name='', args=[], execid=[], range={} or () or var, output_halo=..., modifier_dict={} )'
		output = ''
		
		func_name, idx = get_func_name(line)
		args = get_args(line[idx:])
		execid = get_execid(line, idx)
#				modifier_dict = get_modifier_dict(line, idx)
		
		b_idx = line.find(func_name)
		before = line[:b_idx]
	
		output += before
		output += 'run_function('
		# output name
		return_name = get_return_name(line, idx-len(func_name)-1)
		output += 'return_name=\'' + return_name + '\''

		output += ', func_name=\'' + func_name + '\''
		# args
		
		output += ', args=['
		tp = as_list(args)
		for elem in tp:
			if elem in AXIS:
				output += "'" + elem + "'"
			else:
				output += elem
			output += ', '
		output += ']'
		
		#output += 'args=[' + str(args) + ']'
		#print "O", output
		# arg names
		output += ', arg_names=' + str(as_list(args))
		output += ', execid=' + execid
		
		# range
		range_modifier, flag = get_range(line, idx)
		if flag:
			output += ', work_range=' + as_python(range_modifier)
			
		# output halo
		output_halo, flag = get_output_halo(line, args)
		if flag:
			output += ', output_halo=' + output_halo
		
		# split
		split, flag = get_split(line, idx)
		if flag:
			output += ', split_dict=' + as_python(split)
		
		# merge
		merge_func, merge_order, flag = get_merge(line, idx)
		if flag:
			output += ', merge_func=' + merge_func + ', merge_order=' + merge_order

		# halo
		halo, flag = get_halo(line, args)
		if flag:
			output += ', halo_dict=' + as_python(halo)
		
		# dtype dict
		dtype_dict, flag = get_dtype(line, args)
		if flag:
			output += ', dtype_dict=' + as_python(dtype_dict)

		#output += ', modifier_dict=' + modifier_dict
#				output += ')\n'
		output += ')'
		left = get_left(line, idx)
		output += left
		
		return output
	
	# find VIVALDI_FUNCTION
	while i < n:
		w = code[i]

		line += w
		# find modifier 
		if m_flag == False and i+2 < n: # not found modifier yet
			two_w = code[i:i+2]
			if two_w == ').' and i+2 < n:
				for modifier in modifier_list:
					if code[i+2:].startswith(modifier):
						m_flag = True # found modifier
						break
		if w == '\n':
			if m_flag: # there are modifier in this line
				output += change_line(line)
				#output += '\n'
			else: # no modifier in this line
				output += line
			line = ''
			m_flag = False
		i += 1
	
	if line != '':
		if m_flag:
			output += change_line(line)
		else: 
			output += line
	return output
	
def add_VIVALDI_WRITE(code):
	# this function add VIVALDI_WRITE function call 
	# where assignment operator is
	
	# exception is save_image or save_data

	# initialization
	pair_list = [('[',']'), ('{','}'), ('(',')'), ('"','"'), ("'","'")]
	ip_flag = False
	vw_flag = False
	n = len(code)
	i = 0
	output = ''
	symbol = ''
	var_name = ''
	
	line = ''
	while i < n:
		w = code[i]
		line += w
		if w == '\n':
			flag = False
			m = len(line)
			j = 0
			while j < m:
				w = line[j]
				# skip string
				flag, j = skip_string(line, j)
				if flag: 
					continue
				
				# skip parenthesis
				flag, j = skip_parenthesis(line, j)
				if flag:
					continue
					
				# find assignment
				if w == '=' and (j-1 >= 0 and line[j-1] != '=') and (j+1 < m and line[j+1] != '='):
					flag = True
					break
				j += 1
				
			if flag:
				before = line[:j]
				after = line[j+1:].rstrip()
				var_name = before.strip()
				
				output += before + '= VIVALDI_WRITE(\'' + var_name + '\',' + after + ')\n'
				
				
			else:
				output += line
				
			line = ''
		i += 1
	output += line	
	return output
	
def find_memory_access(line, object_list, level):
	# find is there memory access in the line
	# using recursive way
	
	elem_list = divide_line(line)
	i = 0
	n = len(elem_list)
	flag = False
	
	if level == 0:
		# find direct assignment
		while i < n:
			elem = elem_list[i]
			if elem == '=':
				flag = True
				i += 1
				break
			i += 1
			
		start = i-1
	
	if flag == False: # no direct assignment
		i = 0
		
	while i < n: # memory access checking
		elem = elem_list[i]
		
		# recursive when parenthesis case
		if len(elem) > 2 and elem[0] in ['[','(','{']:
			flag, var_name = find_memory_access(elem[1:-1], object_list, level+1)
			if flag:
				return True, var_name
				
		if elem in object_list: # check element
			return True, elem
		i += 1
	
	return False, ''
	
def find_memory_access_with_exception(line, object_list, level):
	# find is there memory access in the line
	# using recursive way
	
	elem_list = divide_line(line)
	i = 0
	n = len(elem_list)
	flag = False
	
	if level == 0:
		# find direct assignment
		while i < n:
			elem = elem_list[i]
			if elem == '=':
				flag = True
				i += 1
				break
			i += 1
			
		start = i-1
	
	if flag == False: # no direct assignment
		i = 0
		
	while i < n: # memory access checking
		elem = elem_list[i]
		
		
		# exception handling
		if elem in ['save_image','save_data','run_function']:
			return False, ''
		
		# recursive when parenthesis case
		if len(elem) > 2 and elem[0] in ['[','(','{']:
			flag, var_name = find_memory_access_with_exception(elem[1:-1], object_list, level+1)
			if flag:
				return True, var_name
				
		if elem in object_list: # check element
			return True, elem
		i += 1
	
	return False, ''
	
def add_VIVALDI_GATHER(code):
	# Add VIVALDI GATHER to proper place
	
	# problem definition:
	# volume can be distributed in several different physical memory
	# and sometime the volume is necessary in main manager
	# so we will add gather function in the main manager
	
	# when?
	# if there are memory access to volume
	# exception is save_image and input argument of 'run_function'
	
	# initialization
	output = ''
	line_list = code.split('\n')
	object_list = []
	i = 0
	n = len(line_list)
	
	# implementation
	while i < n:
		line = line_list[i]
		output += line
		# check left operand is VIVALDI object
		elem_list = divide_line(line)
		if '=' in elem_list: # exception handling
			idx = elem_list.index('=')
			left = ''
			for a in elem_list[:idx]:
				left += a
			right = ''
			for b in elem_list[idx+1:]:
				right += b
			
			# 1. run_function
			flag = right.startswith('run_function')
			if flag and left not in object_list:
				object_list.append(left)
				
			# 2. pointer direct assignment
			flag = right in object_list
			if flag and left not in object_list:
				object_list.append(left)
				
			# 3. out_of_core load_data, domain specific function. Out of core, but why want to gather?
			# I don't need implement here, because It will be error
			pass
			
		# check there are memory access in the next line
		if i + 1 == n: # this is last line
			i += 1
			continue
	
		# find memory access
		# how can we handle exception??
#		flag, var_name = find_memory_access(line_list[i+1], object_list, 0)
		flag, var_name = find_memory_access_with_exception(line_list[i+1], object_list, 0)
		
		if flag:
			indent = get_indent(line_list[i+1])
			output += '\n' + indent + var_name + ' = ' + 'VIVALDI_GATHER(' + var_name + ')'
			
		output += '\n'
		i += 1
 
	return output

def find_function_name(code):
	# initialization
	############################################################
	output = ''
	n = len(code)
	st = 'func_name='
	cnt = 0
	# implementation
	############################################################
	i = code.find(st) + len(st)
	while i < n:
		w = code[i]
		if w in ['[','{']:
			cnt += 1
		if w in [']','}']:
			cnt -= 1
		if w == ',':
			if cnt == 0:
				break
		output += w	
		i += 1
	
	# some value is evaluable
	# and variable name is not
	try:
		output = ast.literal_eval(output)
	except:
		output = output
	return output

def find_argument(code):
	# initialization
	############################################################
	output = ''
	n = len(code)
	st = 'args='
	cnt = 0
	# implementation
	############################################################
	i = code.find(st) + len(st)
	while i < n:
		w = code[i]
		if w in ['[','{']:
			cnt += 1
		if w in [']','}']:
			cnt -= 1
		if w == ',':
			if cnt == 0:
				break
		output += w	
		i += 1
	
	# some value is evaluable
	# and variable name is not
	try:
		output = ast.literal_eval(output)
	except:
		output = output
	return output

def find_execid(code):
	# initialization
	############################################################
	output = ''
	n = len(code)
	st = 'execid='
	cnt = 0
	# implementation
	############################################################
	i = code.find(st) + len(st)
	while i < n:
		w = code[i]
		if w in ['[','{']:
			cnt += 1
		if w in [']','}']:
			cnt -= 1
		if w == ',':
			if cnt == 0:
				break
		output += w	
		i += 1
	
	# some value is evaluable
	# and variable name is not
	try:
		output = ast.literal_eval(output)
	except:
		output = output
	return output
	
def find_modifier_dict(code):
	# initialization
	############################################################
	output = ''
	n = len(code)
	st = 'modifier_dict='
	cnt = 0
	# implementation
	############################################################
	i = code.find(st) + len(st)
	while i < n:
		w = code[i]
		if w in ['[','{']:
			cnt += 1
		if w in [']','}']:
			cnt -= 1
		if w == ',':
			if cnt == 0:
				break
		output += w	
		i += 1
	
	#print output
	output = output[:-1]
	# some value is evaluable
	# and variable name is not
	try:
		output = ast.literal_eval(output)
	except:
		output = output
	return output

def redundancy_check(input, func_list):
	# redundancy check
	func_name = input['func_name']
	dtype_dict = input['dtype_dict']
	flag = True
	for elem in func_list:
		if func_name == elem['func_name'] and dtype_dict == elem['dtype_dict']:
			flag = False
			break
	return flag
		
def parse_function_call(f_call):
	elem_list = ['func_name=','execid=','args=','dtype_dict=','halo_dict=','work_range=','output_halo=','split_dict=', 'merge_func=', 'merge_order=']
	temp = {}
	
	for elem in elem_list:
		key = elem
		s_idx = f_call.find(key)
		if s_idx == -1:continue
		flag, e_idx = skip_parenthesis(f_call, s_idx+len(key))

		if s_idx+len(key) == e_idx:
			m = len(f_call)
			while e_idx < m:
				w = f_call[e_idx]
				if w in [',',')']:
					break
				e_idx += 1

		value = f_call[s_idx+len(key):e_idx]
		if len(value) != 0:
			try:
				temp[key[:-1]] = ast.literal_eval(value)
			except:
				temp[key[:-1]] = value
				
	return temp
		
def make_merge_function_list(chan_function_list):
	output_list = []
	chan_function_list = [chan_function_list]
	for elem in chan_function_list:
		if 'merge_func' in elem: # merge function exist
			temp = {}
			temp['func_name'] = elem['merge_func']
			temp['merge_order'] = elem['merge_order']
			temp['merge_flag'] = True
			temp['user_function'] = elem['func_name']

			output_list.append(temp)
	return output_list
	
def get_function_list(code):
	
	# find function and merge functions in the code
	
	# initialization
	############################################################
	n = len(code)
	line = ''
	i = 0
	ip_flag = False
	pair_list = [('"','"'), ("'","'")]
	vf = 'run_function'
	function_list = []
	merge_function_list = []
	# implementation
	############################################################

	idx = -1
	while True:
		idx = code.find(vf, idx+1)
		if idx == -1:break

		start = idx + len(vf)
		flag, end =	skip_parenthesis(code, start)
		f_call = code[start:end]

		# parsing function call
		temp = parse_function_call(f_call)
		function_list.append(temp)
		
		# make merge functions
		temp_merge_function_list = make_merge_function_list(temp)
		
		for new_elem in temp_merge_function_list:
			flag = True
			for elem in merge_function_list:
				if elem['func_name'] == new_elem['func_name']:
					flag = False
					break
			if flag:
				merge_function_list += [new_elem]
		
	return function_list, merge_function_list
	
def make_dtype_dict(func_list, merge_function_list):

	# make dtype dictionary 
	n = len(func_list)
	for i in range(n):
		func = func_list[i]
		dtype_dict = func['dtype_dict'] if 'dtype_dict' in func else {}
		arg_list = func['args']
		
		# there are n case for data dtypes
		# 1. we know from dtype modifier
		# 2. axis [x,y,z,w]
		# 3. float 1, 3, 5.7 digit
		# 4. Unknown, volume or value
		
		for arg in arg_list:
			print "AG", arg
			if arg in dtype_dict: # known dtype from dtype modifier 
				dtype_dict[arg] = dtype_dict[arg]
			elif arg in AXIS: # axis 
				dtype_dict[arg] = 'integer'
			elif arg.isdigit(): # digit 
				dtype_dict[arg] = 'float'
			else: # variable maybe volume or value
				dtype_dict[arg] = 'Unknown'
	
		func_list[i]['dtype_dict'] = dtype_dict
		i += 1
		
	return func_list, merge_function_list

# Parse main
#######################################################
# parsing VIVALDI main function to python main function
def parse_main(code):

	# Change modifier
	#
	#######################################################
	code = change_modifier(code)
	
	# ADD VIVALDI_GATHER
	#
	#######################################################
	code = add_VIVALDI_GATHER(code)
	
	# Get function list
	# ADD VIVALDI_WRITE
	#
	#######################################################
	code = add_VIVALDI_WRITE(code)
	
	#
	#######################################################
	func_list, merge_function_list = get_function_list(code)
	
	# Matching 
	#
	#######################################################
# 	dtype dict and merge_function_list not used any more from interactive mode start
#	func_list, merge_function_list = make_dtype_dict(func_list, merge_function_list)
	
	return code, func_list, merge_function_list

def test(test_data, test_set=True, detail=0):
 
	if test_set:
		from preprocessing.main import preprocessing
		
		test_input = test_data['test_input']
		# preprocessing test input
		test_input = preprocessing(test_input)
		
		test_output = test_data['test_output']

	#	result, func_list = parse_main(test_input)
		result, func_list, merge_function_list = parse_main(test_input)

		flag = dif(result.rstrip(), test_output.rstrip())
		
		if flag:
			print "OK"
			if detail >= 1: 
				print result
			return True
		else:
			print "FAILED"
			print "test_input:", test_input
			print "test_output:", test_output
			print "result:", result
			print "end line check"
			return False
	else:
		test_input = test_data
		from preprocessing.main import preprocessing
		test_input = preprocessing(test_input)
		
		main_code = find_main(test_input)
		
		main_code, output, merge_function_list = parse_main(main_code)
		print "MAIN CODE"
		print "========================================"
		print main_code
		print "OUTPUT"
		print "========================================"
		for elem in output:
			print "NAME", elem
		print "MERGE FUNCTION LIST"
		print "========================================"
		print merge_function_list
		# ..
