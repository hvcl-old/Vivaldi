# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
sys.path.append(VIVALDI_PATH+'/src')
from common import *
#####################################################################

from Vivaldi_misc import *
from general.general import *

class Vivaldi_function(object):
	def __init__(self, Vivaldi_code, log_type=log_type, func_name=None, func_args=None, func_dtypes=None):
		pass

class Vivaldi_functions:
	"""
		'Vivaldi_functions' involves 'Vivaldi_function' Objects as list 
	"""
	def __init__(self, Vivaldi_code, log_type):
		self.log_type = log_type

		from vivaldi_translator import translator
		mc, output, globals  = translator(Vivaldi_code, 'CUDA')
		
		from preprocessing.main import preprocessing
		Vivaldi_code = preprocessing(Vivaldi_code)
		self.python_code_dict = self.make_python_code_dict(Vivaldi_code)
		
		self.mc = mc
		self.output = output
		self.globals = globals

	# common
	# CUDA
		
	def test_cuda_compile(self, code):
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

	def get_return_dtype(self, function_name):
		if function_name in self.output:
			dtype = self.output[function_name]['return_dtype']
		else:
			print "VIVALDI Bug"
			print "-----------------------"
			print "Problem occur during prepare function"
			print "Function want to use:", function_name
			print "Prepared:", self.output.keys()
			print "-----------------------"
			assert(False)
		if dtype == 'Unknown': dtype='float'
		return dtype
