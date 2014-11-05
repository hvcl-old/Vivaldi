# import common module
import os, sys
Vivaldi_path = os.environ.get('vivaldi_path')
sys.path.append(Vivaldi_path+'/src')
from common import *
#####################################################################

def get_return_dtype(function_name, argument_package_list, function_code):
	from vivaldi_translator import parse_function
	return_dtype = parse_function(function_code, function_name, argument_package_list)
	if return_dtype == '':
		print "Vivaldi system error"
		print "Wrong return dtype"
		print "dtype:", return_dtype
		assert(False)
	return return_dtype
def line_translator(line, data_package_list):
	def modifier_translator(line):
		from translator.parse_main.main import change_modifier
		return change_modifier(line)
	# translate modifier
	line = modifier_translator(line)
	# translate gather
	def gather_trnslator(line):
		#print data_package_list
		return line
	line = gather_trnslator(line)
	return line
def parse_main(code):
	from translator.parse_main.main import parse_main
	code,_,_ = parse_main(code)
	return code
def preprocessing(code):
	from preprocessing.main import preprocessing
	code = preprocessing(code)
	return code
	
#from Vivaldi_misc import *
#from general.general import *
"""
class Vivaldi_translator_layer:
	
		'Vivaldi_functions' involves 'Vivaldi_function' Objects as list 
		
	def __init__(self):
		#self.log_type = log_type
				
		#self.set_Vivaldi_code(Vivaldi_code)
		#self.set_translated_main_code(Vivaldi_code)
		
		
		from vivaldi_translator import translator
		mc, output, globals  = translator(Vivaldi_code, 'CUDA')
		
		self.python_code_dict = self.make_python_code_dict(Vivaldi_code)
		
		self.mc = mc
		self.output = output
		self.globals = globals
		
		#self.globals = ''
	# common
	
	def get_return_dtype(function_name, argument_package_list, function_code):
		from vivaldi_translator import parse_function
		dtype_dict = [] # what do you want?
		
		return_dtype = parse_function(function_code, function_name, dtype_dict)
		return return_dtype
	
	# CUDA
	def set_Vivaldi_code(self, Vivaldi_code):
		from preprocessing.main import preprocessing
		self.Vivaldi_code = preprocessing(Vivaldi_code)
		
	def set_translated_main_code(self, Vivaldi_code):
		from vivaldi_translator import translate_main
		self.mc = translate_main(Vivaldi_code)
		
#	def get_return_dtype(self, func_name, dtype_dict):
#		from vivaldi_translator import parse_function
#		return_dtype = parse_function(self.Vivaldi_code, func_name, dtype_dict)
#		return return_dtype
		
	def get_function_argument_list(self, func_name):
		func_code = find_code(func_name, self.Vivaldi_code)
		function_argument_list = get_argument(func_code)
		return function_argument_list
		
	def test_cuda_compile(self, code):
		main = 
				int main()
				{
					    return EXIT_SUCCESS;
				}
				

		test_code = attachment + '\n' + code + '\n' + main
		
		f = open('tmp_compile.cu','w')
		f.write(test_code)
		f.close()

		e = os.system('nvcc tmp_compile.cu')
		if e != 0:
			return False
		return True

	def get_cuda_code(self):
		output = 'extern "C"{\n'
		for elem in self.output:
			code = self.output[elem]['code']
			flag = self.test_cuda_compile(code)
			# check
			if flag:
				output += code

		output += '}'
		return output

	def get_attachment(self):
		return attachment

	# Parallel Python
	# Python

	def get_python_code_dict(self):
		return self.python_code_dict
	
	def get_function_list(self, code):
		output_list = []
		start = -1
		end = 0
		while True:
			start = code.find('def ', end)
			if start == -1:
				break
			end = code.find('(', start)
			function_name = code[start+4: end]
			
			if function_name not in output_list:
				output_list.append(function_name)
		
		
		# main is special case
		if 'main' in output_list:
			output_list.remove('main')
			
		return output_list
	
	def make_python_code_dict(self, code):
		output_dict = {}
		
		function_name_list = self.get_function_list(code)
		for function_name in function_name_list:
			function_code = find_code(function_name, code)
			output_dict[function_name] = function_code
			
		return output_dict

	def get_python_code(self, key):
		code = self.python_code_dict[key]
		return code

	# Vivaldi
	def keys(self):
		return self.output.keys()

	def get_Vivaldi_code(self, key):
		return ''
	
	def get_front(self):
		return self.globals

	def get_main(self):
		output = ''
		i = 0
		line_list = self.mc.split('\n')
		n = len(line_list)
		while i < n:
			line = line_list[i]
			if line.startswith('def'):
				i += 1
				continue
		#	output += line[1:]
			output += line[4:]
			if i+1 < n:
				output += '\n'
			i +=1
		return output

	def get_function_name(self):
		return output

"""