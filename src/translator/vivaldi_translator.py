# this is Vivaldi compiler
# change Vivaldi code to several language for each device

import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
path = VIVALDI_PATH+'/src'
if path not in sys.path:
	sys.path.append(VIVALDI_PATH+'/src')
	
from common_in_translator import *
from general.divide_line.divide_line import divide_line

def find_globals(code):
	output = ''
	line_list = code.split('\n')
	i = 0
	n = len(line_list)
	c_indent = ''
	skip = False
	while i < n:
		line = line_list[i]
		indent = get_indent(line)
		
		if indent == '':
			if line.strip().startswith('def'):
				skip = True
			elif len(line.strip()) > 0:
				skip = False
		
		if skip == False:
			output += line
			if i + 1 == n:
				i += 1
				continue
			output += '\n'
		i += 1
		
	return output
	
def append_with_redandant_check(output_merge_function_list, input, check_list):
	# redundant check before add to merge function
	for elem in output_merge_function_list:
		key_flag = True
		for key in check_list:
			if input[key] != elem[key]:
				key_flag = False
				break
		if key_flag:
			return
			
	output_merge_function_list.append(dict(input))
	return
	
def make_CUDA_kernel_func_name(func_name, function_argument_list, dtype_dict):
	for arg in function_argument_list:
		if arg in dtype_dict: 
			if arg in AXIS:continue
			dtype = dtype_dict[arg]
			if dtype.endswith('_volume'): # only volume change CUDA kernel name
				dtype = dtype.replace('_volume','')
	
				func_name += dtype
			
	return func_name
	
def translate_to_CUDA_func_list(func_list, input_merge_function_list):
	output = {}
	merge_function_list = []
	
	for func in func_list:
		func_name = func['func_name']
		o_func_name = str(func_name)
		func_code = func['code']		
		if 'dtype_dict' in func: dtype_dict = func['dtype_dict']
		else: dtype_dict = {'x':'integer', 'y':'integer'}

		# translation
		from vi2cu_translator.main import vi2cu_translator
		
		# argument matching
		function_call_argument_list = func['args']
		function_argument_list = get_argument(func_code)
		dtype_dict = dtype_matching(function_call_argument_list, function_argument_list, dtype_dict)
	
		 # translate
		code, return_dtype = vi2cu_translator(func_code, dtype_dict)
	
		# make CUDA function name
		func_name = make_CUDA_kernel_func_name(func_name, function_argument_list, dtype_dict)
	
		output[func_name] = {'code':code,'return_dtype':return_dtype}

		# merge function
		for m_elem in input_merge_function_list:
			if m_elem['user_function'] == o_func_name:
				m_elem['args'] = ['front','back','x','y'] # need real function
				m_elem['dtype_dict'] = {}
				m_elem['dtype_dict']['front'] = return_dtype + '_volume'
				m_elem['dtype_dict']['back'] = return_dtype + '_volume'

				# append with redundant check
				# id is front, back dtype and function name
				#append_with_redandant_check(merge_function_list, m_elem, ['func_name', 'args', 'dtype_dict'])
				append_with_redandant_check(merge_function_list, m_elem, m_elem.keys())

	return output, merge_function_list
"""
def translate_to_CUDA(func_list, input_merge_function_list, code):
	full_code = str(code)
	i = 0
	n = len(func_list)
	while i < n: # find function code
		func = func_list[i] 
		func_name = func['func_name']
		
		func['code'] = find_code(func_name, full_code)
		i += 1	
	
	output, merge_function_list = translate_to_CUDA_func_list(func_list, input_merge_function_list)
	
	n = len(merge_function_list)
	
	i = 0
	while i < n:
		func = merge_function_list[i] 
		func_name = func['func_name']
		func['code'] = find_code(func_name, full_code)
		i += 1	
	
	for func in merge_function_list:
		func_name = func['func_name']
		func_code = func['code']
		dtype_dict = func['dtype_dict']

		# redundant check
		arg_list = func['args']
		for arg in arg_list:
			if arg in dtype_dict:
				dtype = dtype_dict[arg]
				dtype = dtype.replace('_volume','')
				func_name += dtype
	
		if func_name in output:
			continue

		from vi2cu_translator.main import vi2cu_translator
		code, return_dtype = vi2cu_translator(func_code, dtype_dict)
		output[func_name] = {'code':code,'return_dtype':return_dtype}
	
	
	return output
"""
def chan_temp(dtype_dict, temp, keys, p, final_list):
	n = len(dtype_dict)
	if p >= n:
		import copy
		final_list.append(copy.deepcopy(temp))
		return temp
	
	key = keys[p]
	dtype = dtype_dict[key]
	if dtype.endswith('_volume'):
		for i in ['','2','3','4']:
			front = dtype.replace('_volume', '')
			back = '_volume'
			temp[key] = front + i + back
			chan_temp(dtype_dict, temp, keys, p+1, final_list)
	else:
		temp[key] = dtype
		chan_temp(dtype_dict, temp, keys, p+1, final_list)
		
	return temp
	
def duplicate_function_list(func_list):

	temp_list = []
	for func in func_list:
		dtype_dict = func['dtype_dict']
		keys = dtype_dict.keys()
		final_list = []
		chan_temp(dtype_dict, {}, keys, 0, final_list)
		for dtype_dict in final_list:
			import copy
			aaa = copy.deepcopy(func)
			aaa['dtype_dict'] = dtype_dict
			# Need redundant check later
			temp_list.append(aaa)
			
	return temp_list
	
# VIVALDI translator
#######################################################
def translate_main(code):
	# initialization
	###################################################
	from preprocessing.main import preprocessing
	from parse_main.main import parse_main
	# Preprocessing
	#
	###################################################
	code = preprocessing(code)
	
	# parse main
	#
	###################################################
	mc = find_main(code)
	mc, _, _ = parse_main(mc)
	
	return mc
	
def parse_function(function_code, function_name, argument_package_list):
	from vi2cu_translator.main import vivaldi_parser

	 # translate
	code, return_dtype = vivaldi_parser(function_code, argument_package_list)
	
	return return_dtype
	
def translate_to_CUDA(Vivaldi_code='', function_name='', function_arguments=''):
	from vi2cu_translator.main import vi2cu_translator
	import numpy
	func_code = find_code(function_name=function_name, code=Vivaldi_code)
	
	function_argument_list = get_argument(func_code)
	dtype_dict = {}
	
	i = 0
	for dp in function_arguments:
		data_name = dp.data_name
		f_data_name = function_argument_list[i]
		dtype = dp.data_contents_dtype
		if dp.data_dtype == numpy.ndarray:
			dtype += '_volume'
		
		dtype_dict[f_data_name] = dtype
		i += 1

	# check '[' are exist
	idx = func_code.find('[')
	if idx != -1:
		print "Can not use [ in the Vivaldi, please use xxx_query_function"
		exit()

	# translate
	code, _ = vi2cu_translator(func_code, dtype_dict)

	return code
	
def translator(code, target):
	# initialization
	###################################################
	from preprocessing.main import preprocessing
	from parse_main.main import parse_main
	output = {}
	
	# implementation
	###################################################
	
	# Preprocessing
	#
	###################################################
	code = preprocessing(code)
	
	# find globals
	#
	###################################################
	globals = find_globals(code)
	
	# parse main
	#
	###################################################
	mc = find_main(code)
	mc, func_list, merge_function_list = parse_main(mc)
	
	# Channel
	#
	###################################################
	func_list = duplicate_function_list(func_list)
	
	# Target Translator
	#
	####################################################
	# done list
	done_list = []
	# translate
	target = target.lower()
	if target.strip() == 'cuda':
		output = translate_to_CUDA(func_list, merge_function_list, code)		
	elif target.strip() == 'python':
		pass
	else:
		# debug
		print "NOT proper target"
		print "A", target, "B"
		assert(False)
	
	return mc, output, globals
	assert(False)
	
def print_test_output(main_code, output, globals):
	result = ''
	# print main
	result += '\n'
	result += "* global variables"
	result += '\n'
	result += "******************************************"
	result += globals
	result += '\n'
	result += "* main"
	result += '\n'
	result += "******************************************"
	result += '\n'
	result += main_code
	
	if len(output) > 0:
		result += '\n'
	
	# print etc
	for elem in output:
		result += "* " +  elem
		result += '\n'
		result += "******************************************"
		result += '\n'
		result += output[elem]['code']
		result += '\n'
		result += "return dtype: " + output[elem]['return_dtype']
		result += '\n'
		
	return result
	
def select_compilable(function_dict):
	output_dict = {}
	for elem in function_dict:
		flag = test_cuda_compile(function_dict[elem]['code'])
		if flag == True:
			output_dict[elem] = function_dict[elem]
			
	return output_dict
		
def test(test_data, test_set=True, detail=0):

	if test_set:
		test_input = test_data['test_input']
		target = test_data['target']
		test_output = test_data['test_output']
		
		main_code, output, globals = translator(test_input, target)
		output = select_compilable(output)
		result = print_test_output(main_code, output, globals)
		flag = dif(result, test_output)
		
		if flag:
			print "OK"
			if detail > 0: print result
			return True
		else:
			print "FAILED"
			print "test_input:", test_input
			print "test_output:", test_output
			print "result:", result
			print "END_LINE"
			return False
	else:
		test_input = test_data
		target = 'CUDA'
		
		main_code, output, globals = translator(test_input, target)
		output = select_compilable(output)
		result = print_test_output(main_code, output, globals)
		print result
		# ..
		return True
	return False
