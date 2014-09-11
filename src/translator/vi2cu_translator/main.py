import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
VIVALDI_PATH = '/home/hschoi/Vivaldi'
path = VIVALDI_PATH+'/src/translator'
if path not in sys.path:
	sys.path.append(path)
	
from common_in_translator import *
from general.divide_line.divide_line import divide_line
from functions.split_into_block_and_code.split_into_block_and_code import split_into_block_and_code
from functions.dtype_check.dtype_check import find_dtype
	
def get_CUDA_dtype(elem, local_dict, type):

	# default dtype is float
	dtype = 'float'
	
	
	# data type to CUDA dtype
	if elem in local_dict:
		dtype = local_dict[elem]
		
		if type == 'func_name':
			if dtype.endswith('_volume'):
				dtype = dtype.replace('_volume','')
			if dtype == 'integer':
				dtype = dtype.replace('integer','int')
		elif type == 'args':
			if dtype.endswith('_volume'):
				dtype = dtype.replace('_volume','*')
			
			if dtype == 'integer':
				dtype = dtype.replace('integer','int')	
		elif type == 'body':
			#if dtype.endswith('_volume'):
			#	dtype = dtype.replace('_volume','*')
			
			if dtype == 'integer':
				dtype = dtype.replace('integer','int')
		elif dtype == 'Unknown':
			dtype = 'float'
		dtype = dtype.replace('Unknown', 'float')	
		
	return dtype
	
# find out dtype and edit the line to CUDA line.
def parse_line(line, local_dict):

	# Find data type
	#
	#####################################################################################
	# find data type of elements in the line
	from vi2cu_translator.functions.dtype_check.dtype_check import find_dtype
	find_dtype(line, local_dict)

# find possible dtypes of each variables,
# and change functions to CUDA style of code
def parse_code(code, local_dict):
	# parse code line by line

	# Split to line list
	#
	#####################################################################################
	# split code to multiple lines
	line_list = code.split('\n')

	
	# Parse line
	#
	#####################################################################################
	# parse line by line
	for line in line_list:
		parse_line(line, local_dict)

def get_indent(line):
	s_line = line.strip()
	i_idx = line.find(s_line)
	indent = line[:i_idx]

	return indent
	
def add_semicolon(block):
	# attach semicolon with correct end line
	n = len(block)
	output = ''
	line= ''
	for i in range(n):
		if block[i] == '\n':
			if i-1 > 0 and block[i-1] not in ['{','}'] and line.strip() != '':
				output += ';'
			line = ''
		output += block[i]
		line += block[i]
	return output
	
def change_if(code_list, local_dict):
	# initialization
	###################################################
	output_list = []
	string_symbol = ['"',"'"]
	s_symbol = ''
	s_flag = False
	
	pair_list = [('[',']'), ('{','}'), ('(',')')]
	p_symbol = ''
	p_cnt = 0
	p_flag = False
	# implementation
	###################################################
	for code in code_list:
		n = len(code)
		i = 0
		output = ''
		while i < n:
			w = code[i]
			if s_flag: # in the string
				if w == st:
					s_flag = False
					
				output += w
			elif p_flag: # in the parenthesis
			
				# check string 
				for st in string_symbol:
					if w == st:
						symbol = w
						s_flag = True
						break
				if s_flag:
					output += w
					continue
				
				if w == p_symbol:
					p_cnt -= 1
					if p_cnt == 0:
						p_symbol = ''
						p_flag = False
				
				output += w
			else: # out of parenthesis
			
				# check string 
				for st in string_symbol:
					if w == st:
						symbol = w
						s_flag = True
						break
				if s_flag:
					output += w
					continue
				
				# check parenthesis
				for pair in pair_list:
					p_st = pair[0]
					if w == p_st:
						p_symbol = pair[1]
						p_cnt += 1
				
				# change to CUDA style
				if code[i-1:i+2] == ' if': # change if to CUDA style
					target = 'if'
					length = len(target)
					output += target + '('
					
					# go to until end line
					i += length
					while True:
						w = code[i]
						if w == ':':
							output += '){\n'
							i += 3
							break
						output += w
						i += 1
						
				elif code[i-1:i+4] == ' elif': # change else if to CUDA style
					output += 'else if('
					
					target = 'elif'
					length = len(target)
					
					# go to until end line
					i += length
					while True:
						w = code[i]
						if w == ':':
							output += '){\n'
							i += 3
							break
						output += w
						i += 1
				elif code[i-1:i+4] == ' else': # change else to CUDA style
					target = 'else'
					length = len(target)
					output += target
					
					# go to until end line
					i += length
					while True:
						w = code[i]
						if w == ':':
							output += '{\n'
							i += 3
							break
						output += w
						i += 1
				else: # no
					output += w
					
			i += 1
		
		output_list.append(output)
		
	return output_list

def change_while(code_list, local_dict):
	# initialization
	###################################################
	output_list = []
	string_symbol = ['"',"'"]
	s_symbol = ''
	s_flag = False
	
	pair_list = [('[',']'), ('{','}'), ('(',')')]
	p_symbol = ''
	p_cnt = 0
	p_flag = False
	# implementation
	###################################################
	for code in code_list:
		n = len(code)
		i = 0
		output = ''
		while i < n:
			w = code[i]
			if s_flag: # in the string
				if w == st:
					s_flag = False
					
				output += w
			elif p_flag: # in the parenthesis
			
				# check string 
				for st in string_symbol:
					if w == st:
						symbol = w
						s_flag = True
						break
				if s_flag:
					output += w
					continue
				
				if w == p_symbol:
					p_cnt -= 1
					if p_cnt == 0:
						p_symbol = ''
						p_flag = False
				
				output += w
			else: # out of parenthesis
			
				# check string 
				for st in string_symbol:
					if w == st:
						symbol = w
						s_flag = True
						break
				if s_flag:
					output += w
					continue
				
				# check parenthesis
				for pair in pair_list:
					p_st = pair[0]
					if w == p_st:
						p_symbol = pair[1]
						p_cnt += 1
				
				# change while to CUDA style
				target = 'while'
				length = len(target)
				if code[i:].startswith(target):
					output += target + '('
					
					# go to until end line
					i += length
					while True:
						w = code[i]
						if w == ':':
							output += '){\n'
							i += 3
							break
						output += w
						i += 1
					
				else:
					output += w
					
			i += 1
		
		output_list.append(output)
		
	return output_list

def change_return(code_list, local_dict):
	# change return statement for CUDA style
	# from return max
	# to rb[(z-rb_DATA_RANGE->start.z)*(rb_DATA_RANGE->end.y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(y-rb_DATA_RANGE->start.y)*(rb_DATA_RANGE->end.x-rb_DATA_RANGE->start.x)+(x-rb_DATA_RANGE->start.x)] = max
	
	k = 0
	for code in code_list:
	
		flag = True
		idx1 = -1
		while flag:
			flag = False
			idx2 = code.find('return', idx1+1)
			
			if idx2 != -1:
				flag = True
				
				# before
				before = code[:idx2]
				# mid
				mid = ''
				AXIS = ['x','y','z','w']
				i = 0
				for axis in AXIS:
					if axis in local_dict:
						mid += '+(' + axis + '-rb_DATA_RANGE->buffer_start.' + axis + ')'
						
						for axis2 in AXIS[:i]:
							mid += '*(rb_DATA_RANGE->buffer_end.' + axis2 + '-rb_DATA_RANGE->buffer_start.' + axis2 + ')'
						
						i += 1
					else:
						break
				mid = 'rb[' + mid[1:] + '] = ' 
				# after
				after = code[idx2 + len('return'):]
				
				idx3 = after.find('\n')
				indent = get_indent(code[:idx2+1])
#				after = after[:idx3+1] + indent + 'return\n' + after[idx3+1:] 
				after = after[:idx3+1].strip() + '\n' + indent + 'return\n' + after[idx3+1:] 
			
				# merge
				code = before + mid + after
				
				# idx1 = found + mid + idx3 + indent
				num = idx2 + len(mid) + idx3 + len(indent)
				idx1 = num + 1
				
		code_list[k] = code
		k += 1
				
	
	return code_list

def change_function(code_list, local_dict):
	# change VIVALDI code to CUDA style
	# this function iterate code list and change one by one
	
	# initialization
	###################################################
	
	# implementation
	###################################################
	k = 0
	for code in code_list:
		p_st = -1
		p_ed = -1
		n = len(code)
		while True:
			# find function
			
			# find parenthesis start
			p_st = code.find('(', p_st+1)
			
			# find parenthesis end
			p_ed = p_st
			p_cnt = 0
			while p_ed < n:
				w = code[p_ed]
				if w == '(': p_cnt += 1
				if w == ')': 
					p_cnt -= 1
					if p_cnt == 0:
						break
				p_ed += 1
				
			if p_st == -1 or p_ed == -1:
				break
			else:
				c_list = []
				# find start of function
				c_list += [' ', '\t', '\n'] # space series
				c_list += ['(', '[' , '{'] # parenthesis series
				c_list += [',',':'] # comma series
				st = p_st-1
				while True:
					if code[st] in c_list: 
						st += 1
						break
					st -= 1
		
				p_ed += 1
				elem = code[st: p_ed]
			
			if is_function(elem):
				func_name, args = split_elem(elem)
				
				# change function name
				before = code[:st]
				
				after = code[p_ed:]
				
				sfn = func_name.strip()
				if sfn in query_list:
					# change function name
					arg_list = divide_line(args)
					f_a = arg_list[0]
					dtype = local_dict[f_a]
					if dtype.endswith('_volume'):
						dtype = dtype[:-7]

					in_type = dtype_convert_in_query(dtype)
					chan = in_type[-1]
					
					# float version
					if chan is 't':	
						dtype = '<float>'
					else: 
						dtype = '<float' + chan + '>'
					func_name += dtype
					p_st += len(dtype)
										
					# change function args
					args += ', ' + f_a + '_DATA_RANGE'
				if sfn in gradient_list:
					# change function name
					arg_list = divide_line(args)
					f_a = arg_list[0]
					dtype = local_dict[f_a]
					if dtype.endswith('_volume'):
						dtype = dtype[:-7]

					in_type = dtype_convert_in_query(dtype)
					chan = in_type[-1]
					
					# float version
					if chan is 't':	
						dtype = '<float>'
					else: 
						dtype = '<float' + chan + '>'
					func_name += dtype
					p_st += len(dtype)
										
					# change function args
					args += ', ' + f_a + '_DATA_RANGE'
				if sfn in ['orthogonal_iter','perspective_iter']:
					# change function args
					arg_list = divide_line(args)
					f_a = arg_list[0]
					args += ', ' + f_a + '_DATA_RANGE'
					

				mid = func_name + '(' + args + ')'
				
				code = before + mid + after
				n = len(code)
			
		code_list[k] = code
		k += 1
		
	
	# split_line method is difficult to recover original code 

	return code_list

def change_WORD_OPERATOR(code_list, local_dict):
	# initialization
	###################################################
	output_list = []
	word_operator_to_change_dict={'and':'&&', 'or':'||', 'not':'!', 'is':'=='}
	# implementation
	###################################################
	for code in code_list:
		n = len(code)
		i = 0
		
		while i < n:
			w = code[i]
			# skip string
			flag, i = skip_string(code, i)
			if flag: 
				continue
				
			# change word operators
			for w_operator in word_operator_to_change_dict:
				L = len(w_operator)
				word = code[i:i+L]
				previous = code[i-1]
				after = 0
				if i + L < n: after = code[i+L]
				if previous not in ['','\n',' ','=']: continue
				if after not in ['', '\n',' ','=']: continue
				if word == w_operator:
					code = code[:i] + word_operator_to_change_dict[w_operator] + code[i+len(w_operator):]
					i += len(w_operator)
					n = len(code)
					break
			i += 1
			
		output_list.append(code)
	return output_list
	
def add_declaration(target, local_dict, in_B, exception=[]):
	output = target
	
	indent = get_indent(output)
	for elem in local_dict:
		# exception handling 
		if elem in exception: continue
		if local_dict[elem] == 'Unknown': continue
		if local_dict[elem].endswith('_constant'): continue
		if local_dict[elem].endswith('_volume'): continue
		if local_dict[elem].endswith('_DATA_RANGE'): continue		
		flag = False
		if elem not in in_B: flag = True
		elif local_dict[elem] != in_B[elem]: flag = True
				
		if flag:
			dtype = get_CUDA_dtype(elem, local_dict, 'body')
			if dtype in ['list','dict','tuple']: continue
			output = indent + dtype + ' ' + elem + '\n' + output

	return output
			
# recursive parsing Vivaldi code
def parse_block(block='', local_dict={}):
	# initialize variables
	use_B = {}
	in_B = dict(local_dict)
	# manage special for statements
	for_flag = False
	s_block = block.strip()
	if s_block.startswith('for'): # for statement is special case
	
		# manage for statement
		#######################################################################################
		idx = block.find('\n')
		line = block[:idx]
		
		# parse variable definition in the for
		elem_list = divide_line(line)
		var_name = elem_list[1]
		dtype = get_dtype(elem_list[3], local_dict, 'for')
		
		add_dtype(var_name, dtype, local_dict)
		
		# remove the for statement line
		for_statement_line = block[:idx+1]
		block = block[idx+1:]
		
		for_flag = True
		
		# Split block and code
		#
		#####################################################################################
		
		# parsing data type
		#####################################################################################	
		
		# block consist of small block and programming code
		# split small blocks and codes	
		block_list, code_list = split_into_block_and_code(block)
		
		output = ''
		n = len(code_list)
		
		for i in range(n):
			# initialize variables
			
			# Parse code
			#
			#####################################################################################
			
			# parsed code until meet next block, and add variable in dictionary
			code = code_list[i]
			parse_code(code, local_dict)

			# Parse block, Recursive step
			#
			#####################################################################################
			if i < len(block_list):
				# parse inner block recursively, we have to give copy of local dictionary
				inner_block = block_list[i]
				parsed_block, inner_dict = parse_block(block=inner_block, local_dict=dict(local_dict))
				
				# update local_dict
				for elem in inner_dict:
					if elem in local_dict and local_dict[elem] == 'Unknown':
						local_dict[elem] = inner_dict[elem]
						
				block_list[i] = parsed_block
		
		# to CUDA style
		#
		####################################################################################
		
		# data type declaration
		if len(code_list) > 0:
			code_list[0] = add_declaration(code_list[0], local_dict, in_B, exception=[var_name])

		# change 'if' to CUDA style
		code_list = change_if(code_list, local_dict)
		# change 'while' to CUDA style
		code_list = change_while(code_list, local_dict)
		# change WORD_OPERATOR to CUDA style
		code_list = change_WORD_OPERATOR(code_list, local_dict)
		
		# change for statement
		if for_flag:
			elem_list = divide_line(for_statement_line)
			indent = get_indent(for_statement_line)
		
			for_statement_line = indent + 'for('
			
			# elem_list[1] is variable
			# elem_list[3~:-1] is iterator-able object
			
			if get_dtype(elem_list[3],local_dict) in ['line_iter','plane_iter','cube_iter']:
				# domain specific language iterator case
				var = elem_list[1]
				iter = elem_list[3]
			
				if var not in in_B:
					for_statement_line += local_dict[var] + ' '
			
				for_statement_line += elem_list[1] + ' = ' + iter + '.begin(); '
				for_statement_line += iter + '.valid(); '
				for_statement_line += '){\n' # increment not come here
			
				if len(code_list) == 0:
					code_list.append('')
				code_list[0] = for_statement_line + code_list[0]
				
				#output = for_statement_line + output
				#output += indent + '\t' + var + ' = ' + var + '.next()\n'
#				code_list.append(indent + '    ' + var + ' = ' + var + '.next()\n')
				code_list.append(indent + '    ' + var + ' = ' + elem_list[3] + '.next()\n')
			else:
				# range case
				
		#		for_statement_line += elem_list[1] + initialize + ';'
		#		for_statement_line += condition + ';'
		#		for_statement_line += '){\n' # increment not come here
			
		#		output = for_statement_line + output
				arg_cnt = elem_list[4].count(',')
				var = elem_list[1]
				if arg_cnt == 1:
					range_values = elem_list[4][1:len(elem_list[4])-1].split(",")
					for_statement_line += "int "+var+"= " + range_values[0] + ";"
					for_statement_line += var+"<" + range_values[1] +";"
					for_statement_line += var+"++){\n"
				elif arg_cnt == 2:
					range_values = elem_list[4][1:len(elem_list[4])-1].split(",")
					for_statement_line += "int "+var+"= " + range_values[0] + ";"
					for_statement_line += var+"<" + range_values[1] +";"
					for_statement_line += var+"="+var+"+(int)"+ range_values[2]+"){\n"
				elif arg_cnt == 0:
					for_statement_line += "int "+var+"=0;"
					for_statement_line += var+"<" + elem_list[4][1:-1] +";"
					for_statement_line += var+"++){\n"
				else:
					print "RANGE IS NOT SUITABLE"
					assert(False)
				#print "RANGE is not implemented yet"
				#print local_dict
				#assert(False)
				code_list[0] = for_statement_line + code_list[0]
		
		# change return
		code_list = change_return(code_list, local_dict)
		# change function
		code_list = change_function(code_list, local_dict)
		# add semicolon
		i = 0
		for code in code_list:
			code_list[i] = add_semicolon(code)
			i += 1
			
		# merge code and blocks
		i = 0
		output = ''
		for code in code_list:
			output += code
			
			if i < len(block_list):
				block = block_list[i]
				output += block
			i += 1
		
		# close block
		if len(code_list) > 0:
			indent = get_indent(output)
			
			output += indent + '}\n'	
	else: # ordinary case
		# Split block and code
		#
		#####################################################################################
		
		# parsing data type
		#####################################################################################
		# block consist of small block and programming code
		# split small blocks and codes	
		block_list, code_list = split_into_block_and_code(block)
		
		output = ''
		n = len(code_list)
		
		for i in range(n):
			# initialize variables

			# Parse code
			#
			#####################################################################################
			# parsed code until meet next block, and add variable in dictionary
			code = code_list[i]
			parse_code(code, local_dict)
			
			# Parse block, Recursive step
			#
			#####################################################################################
			if i < len(block_list):
				# parse inner block recursively, we have to give copy of local dictionary
				inner_block = block_list[i]
				parsed_block, inner_dict = parse_block(block=inner_block, local_dict=dict(local_dict))
				# update local_dict
				for elem in inner_dict:
					if elem in local_dict and local_dict[elem] == 'Unknown':
						local_dict[elem] = inner_dict[elem]
				
				block_list[i] = parsed_block
		
		# to CUDA style
		####################################################################################
		
		# data type declaration
		if len(code_list) > 0:
			code_list[0] = add_declaration(code_list[0], local_dict, in_B, exception=[])
		
		# change 'if' to CUDA style
		code_list = change_if(code_list, local_dict)
		# change 'while' to CUDA style
		code_list = change_while(code_list, local_dict)
		# change WORD_OPERATOR to CUDA style
		code_list = change_WORD_OPERATOR(code_list, local_dict)
		
		# change return
		code_list = change_return(code_list, local_dict)
		# change function
		code_list = change_function(code_list, local_dict)
		# add semicolon
		i = 0
		for code in code_list:
			code_list[i] = add_semicolon(code)
			i += 1
		
		# merge code and blocks
		i = 0
		for code in code_list:
			output += code
			
			if i < len(block_list):
				block = block_list[i]
				output += block
			i += 1
		
		# close block
		if len(code_list) > 0:
			indent = get_indent(output)
			output += indent[:-4] + '}\n'	
		
	return output, local_dict
		
def add_axis_declaration(CUDA_body, local_dict):

	temp = ''
	AXIS = ['x','y','z','w']
	for axis in AXIS:
		if axis in local_dict:
			temp += "\n    int " + axis + "_hschoi = threadIdx." + axis + " + blockDim." + axis + " * blockIdx." + axis + ";"
			temp += '\n    int ' + axis + ' = ' + axis + '_start + ' + axis + '_hschoi;'
	
	temp += '\n'
	
	CUDA_body = temp + CUDA_body
	return CUDA_body
	
def add_boundary_check(CUDA_body, local_dict):

	temp = ''
	temp += '\n    if('
	
	AXIS = ['x','y','z']
	for axis in AXIS:
		if axis in local_dict:
			temp += axis + '_end <= ' + axis
#		else: temp += axis + '_hschoi >= 1'
			temp += ' || '
		
	temp = temp[:-4]
	
	temp += ')return;\n'
	
	CUDA_body = temp + CUDA_body
	return CUDA_body
	
# VI2CU body translator
#######################################################
def parse_body(vivaldi_code_body='', local_dict={}):

	# Parse block
	#
	#######################################################
	# parse Vivaldi code body to CUDA body
	CUDA_body, inner_dict = parse_block(block=vivaldi_code_body, local_dict=local_dict)
	
	# update local_dict
	for elem in inner_dict:
		if elem in local_dict and local_dict[elem] == 'Unknown':
			local_dict[elem] = inner_dict[elem]
			
	# Boundary check
	#
	#######################################################
	# add boundary check
	CUDA_body = add_boundary_check(CUDA_body, local_dict)
	
	# Add axis declaration
	#
	#######################################################
	# add x, y, z to CUDA body
	CUDA_body = add_axis_declaration(CUDA_body, local_dict)
	
	# Find return data type
	#
	#######################################################
	# return compiled code and return data type
	return_dtype = find_return_dtype(CUDA_body, local_dict)

	return CUDA_body, return_dtype

# parse head related functions
############################################################################################
def to_CUDA_head_add_dtype(CUDA_head='', local_dict={}):
	# add dtype to input arguments 
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
		
	output = ''
	output += elem_list[0] + ' ' # def
	output += elem_list[1] # function_name
	
	func_name = elem_list[1]
	args = elem_list[2][1:-1]
	arg_list = divide_line(elem_list[2][1:-1])
			
	if n > 1:
		idx = CUDA_head.find(elem_list[0])
		output += CUDA_head[:idx]
		
	output += '('
	for arg in arg_list:
		if arg in local_dict:
			dtype = get_CUDA_dtype(arg, local_dict, 'args')
			output += dtype + ' ' + arg
		else:
			output += arg + ' '
	output += '):'
	return output

def to_CUDA_head_style(CUDA_head, local_dict):
	# change head style
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
		
	output = ''
	output += '__global__ void ' # def
	output += elem_list[1] # function_name
	
	func_name = elem_list[1]
	args = elem_list[2][1:-1]
	
	output += '('+args+')'
	output += '{'
	return output
	
def change_function_name(CUDA_head, local_dict):
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
	output = ''
	output += elem_list[0] + ' ' # def
	output += elem_list[1] # function_name
	
	args = elem_list[2][1:-1]
	arg_list = divide_line(elem_list[2][1:-1])
	
	for elem in arg_list:
		if elem is ',': continue
		if elem in AXIS: continue
		dtype = get_CUDA_dtype(elem, local_dict, 'func_name')
		if elem in local_dict and local_dict[elem].endswith('_volume'):
			output += dtype

	output += '('+args+')'
	output += ':'
	return output
	
def add_return_buffer(CUDA_head, local_dict):
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
		
	output = ''
	output += elem_list[0] + ' ' # def
	output += elem_list[1] # function_name
	
	args = elem_list[2][1:-1]
	args = 'rb, ' + args
	output += '('+args+')'
	output += ':'
	
	return output
	
def add_axis_in_args(CUDA_head, local_dict):
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
		
	output = ''
	output += elem_list[0] + ' ' # def
	output += elem_list[1] # function_name
	
	func_name = elem_list[1]
	args = elem_list[2][1:-1]
			
	arg_list = divide_line(args)
	args = ''
	AXIS = ['x','y','z','w']
	for arg in arg_list:
		if arg in AXIS:
			local_dict[arg+'_start'] = 'integer'
			local_dict[arg+'_end'] = 'integer'
			args += arg + '_start,'
			args += arg + '_end'
		else:
			args += arg
			if arg == ',':
				args += ' '
	
	output += '('+args+')'
	output += ':'

	return output
	
def add_volume_ranges(CUDA_head, local_dict):
	elem_list = divide_line(CUDA_head)
	n = len(elem_list)
		
	output = ''
	output += elem_list[0] + ' ' # def
	output += elem_list[1] # function_name
	
	if 'rb' in local_dict:
		if not local_dict['rb'].endswith('_volume'):
			local_dict['rb'] = local_dict['rb'] + '_volume'
	
	func_name = elem_list[1]
	args = elem_list[2][1:-1]
			
	arg_list = divide_line(args)
	args = ''
	for arg in arg_list:
		dtype = local_dict[arg] if arg in local_dict else ''
		if dtype.endswith('volume'):
			args += arg
			args += ','
			args += arg +'_DATA_RANGE'
			local_dict[arg+'_DATA_RANGE'] = 'VIVALDI_DATA_RANGE*'
		else:
			args += arg
			if arg == ',':
				args += ' '
			
	
	output += '('+args+')'
	output += ':'
	
	return output
	
def parse_head(vivaldi_code_head='', local_dict={}):

	CUDA_head = vivaldi_code_head
	# Change function name
	#
	#######################################################
	
	CUDA_head = change_function_name(CUDA_head, local_dict)
	# Add return buffer
	#
	####################################################### 
	
	CUDA_head = add_return_buffer(CUDA_head, local_dict)
	
	# Add volume ranges
	#
	#######################################################
	# add volume 'VIVALDI_DATA_RANGE*' for each volume
	
	CUDA_head = add_volume_ranges(CUDA_head, local_dict)
	# Add axis arguments
	#
	#######################################################
	
	CUDA_head = add_axis_in_args(CUDA_head, local_dict)
	# Add data type definition
	#
	#######################################################
	
	CUDA_head = to_CUDA_head_add_dtype(CUDA_head, local_dict)
	# Change to CUDA style
	#
	#######################################################
	# change to CUDA style like '__global__','void','{'
	
	CUDA_head = to_CUDA_head_style(CUDA_head, local_dict)
	
	
	return CUDA_head
		
def split_head_and_body(vivaldi_code=''):

	parenthesis_right = vivaldi_code.find(')')
	
	if parenthesis_right == '-1':
		# where is head?
		assert(False)
		
	end_line = vivaldi_code.find('\n',parenthesis_right)+1
	
	vivaldi_code_head = vivaldi_code[:end_line]
	vivaldi_code_body = vivaldi_code[end_line:]
	
	
	if False:
		# debug code 
		print "Head"
		print vivaldi_code_head
	
		print "Body"
		print vivaldi_code_body

		exit()
	
	return vivaldi_code_head, vivaldi_code_body
	
def to_CUDA_dtype(dtype):
	# remove constant
	dtype = dtype.replace('_constant','')
	
	# replace integer to int
	dtype = dtype.replace('integer', 'int')
	
	return dtype
	
def find_return_dtype(code='', local_dict={}):
	# find return dtype of CUDA code
	
	if False:
		# False
		print "FIND RETURN", local_dict
	n = len(code)
	idx1 = -1
	return_dtype = ''
	while True:
		# find value assignment of return buffer
		idx1 = code.find('rb[', idx1+1)
		if idx1 == -1: break
		
		# find assignment operator
		idx2 = code.find('=',idx1+1)
		
		# find variable name
		flag = False
		i = idx2+1
		var_name = ''
		
		# find variable start
		start = i+1
		
		# find variable end
		while i < n:
			w = code[i]
			
			# end line
			if w == ';':
				break
			i += 1
		end = i
		# variable name
		var_name = code[start: end]
		
		# CUDA to VIVALDI style
		# because implemented function get VIVALDI code to input 
		
		# remove <...>: not in the python
	
		n = len(var_name)
		output = ''
		i = 0
		flag = True
		while i < n:
			w = var_name[i]
			if w == '<':
				flag = False
				
			if flag:
				output += w
				
			if w == '>':
				flag = True
			i += 1

		return_dtype = find_dtype(output, local_dict)
	
		## to CUDA dtype
		return_dtype = to_CUDA_dtype(return_dtype)
		
		return return_dtype
		
	return return_dtype
		
# Target Translator, CUDA
#######################################################
# main of compiler
# translate a VIVALDI function to a CUDA function
def vi2cu_translator(vivaldi_code='', local_dict={}):
	
	# initialize variables
	#######################################################
	vivaldi_code_head, vivaldi_code_body = split_head_and_body(vivaldi_code=vivaldi_code)

	# VI2CU body translator
	#
	#######################################################
	# parse vivaldi_code body to CUDA

#	print local_dict
#	argument_list = get_argument(vivaldi_code_head)

	#function_call_argument_list = func['args']
	#function_argument_list = get_argument(vivaldi_code)
	#dtype_dict = dtype_matching(function_call_argument_list, function_argument_list, dtype_dict)			
			
#	print local_dict, argument_list
#	print vivaldi_code
#	print "Z",local_dict
	CUDA_body, return_dtype = parse_body(vivaldi_code_body=vivaldi_code_body, local_dict=local_dict)
	
	
	# add return dtype to dictionary
	add_dtype('rb', return_dtype, local_dict)
	
	# VI2CU head translator
	#
	#######################################################
	# parse vivaldi_code head to CUDA
	# head must be changed after body
	# because we don't know data type until parse_body finish
	CUDA_head = parse_head(vivaldi_code_head, local_dict=local_dict)
	
	# Prepare Return
	#
	#######################################################
	# merge parsed head and body
	CUDA_code = CUDA_head + '\n' + CUDA_body
	
	return CUDA_code, return_dtype

def vivaldi_parser(vivaldi_code, argument_package_list): # temp function, it should be moved to parser
	vivaldi_code_head, vivaldi_code_body = split_head_and_body(vivaldi_code=vivaldi_code)
	
	function_argument_list = get_argument(vivaldi_code)
	def get_dtype_dict(function_argument_list, argument_package_list):
		
		n = len(function_argument_list)
		m = len(argument_package_list)
		if n != m:
			print "WARNING"
			print "---------------------"
			print "Function argument number is not matching"		
			print function_call_argument_list
			print function_argument_list
			print "---------------------"
			assert(False)
		
		import numpy
		i = 0
		dtype_dict = {}
		for function_argument in function_argument_list:
			argument_package = argument_package_list[i]
			if argument_package.data_dtype == numpy.ndarray:
				dtype_dict[function_argument] = argument_package.data_contents_dtype+'_volume'
			else:
				dtype_dict[function_argument] = argument_package.data_contents_dtype
			i += 1
		return dtype_dict
		
	local_dict = get_dtype_dict(function_argument_list, argument_package_list)	
		
	CUDA_body, return_dtype = parse_body(vivaldi_code_body=vivaldi_code_body, local_dict=local_dict)
	CUDA_head = parse_head(vivaldi_code_head, local_dict=local_dict)
	
	return None, return_dtype
	
def test(test_data, test_set=True, detail=0):
	
	if test_set:
		vivaldi_code = test_data['test_input']
		local_dict = test_data['dtype_dict']
		import ast
		local_dict= ast.literal_eval(local_dict)
		test_output = test_data['test_output']
		test_return_dtype = test_data['return_dtype']
		
		# test
		result, result_return_dtype = vi2cu_translator(vivaldi_code=vivaldi_code, local_dict=local_dict)
		
		flag = True
		if flag: # check code
			flag = dif(result.strip(), test_output.strip())
		if flag:
			print "OK"
			return True
		else:
			print "FAILED"
			print "test_input:", vivaldi_code.strip()
			print "test_output:", test_output.strip()
			print "result:", result.strip()
			return False
		
	else:
		test_input = test_data
		target = 'CUDA'
		
		from preprocessing.main import preprocessing
		
		code = preprocessing(test_input)
		print "CODE"
		print "=============================================="
		print code
		print "FUNCTION_LIST"
		print "=============================================="
		from parse_main.main import parse_main
		mc = find_main(code)
		mc, func_list, merge_function_list = parse_main(mc)
		
		for elem in func_list:
			print "FUNC", elem
			
		print "TRNALSATED"
		print "=============================================="
		
		full_code = str(code)
		i = 0
		n = len(func_list)
		while i < n:
			func = func_list[i] 
			func_name = func['func_name']
			func['code'] = find_code(func_name, full_code)
			i += 1	
		
		output = {}
		for func in func_list:
			func_name = func['func_name']
			func_code = func['code']
			dtype_dict = func['dtype_dict'] if 'dtype_dict' in func else {}

			o_func_name = str(func_name)
			# redundant check
			arg_list = func['args']
			for arg in arg_list:
				if arg in dtype_dict:
					dtype = dtype_dict[arg]
					dtype = dtype.replace('_volume','')
					func_name += dtype
		
			if func_name in output:
				continue
			
			# translation
			from vi2cu_translator.main import vi2cu_translator
			
			# argument matching
			function_call_argument_list = func['args']
			function_argument_list = get_argument(func_code)
			dtype_dict = dtype_matching(function_call_argument_list, function_argument_list, dtype_dict)

			code, return_dtype = vi2cu_translator(func_code, dtype_dict)
			output[func_name] = {'code':code,'return_dtype':return_dtype}
		
		#result, result_return_dtype = vi2cu_translator(vivaldi_code=result, local_dict=[])
		
		for elem in output:
			print "FUNC_NAME", elem
			print "=============================================="
			print output[elem]['code']
		return True
	return False
