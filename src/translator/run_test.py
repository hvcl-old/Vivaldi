import sys
from misc import read_file
import ast


def get_test_vi2cu_translator(file_name):
	test = {}
	a = read_file(file_name)
	test = ast.literal_eval(a)

	if False:
		# print test for testing, read file
		for test_name in test:
			print test[test_name]
		exit()
	
	return test

def get_test_list(tests):
	# remove comment and empty line
	input_list = tests.split('\n')
	test_list = []
	for elem in input_list:
		line = ''
		
		# comment remove
		for w in elem:
			if w == '#': break
			line += w
			
		# empty line remove
		if line.strip() == '':
			continue
			
		# maybe there are not space before and after at the file name
		test_list.append(line.strip())
	
	return test_list
	
def remove_space_after_line(code):
	output = ''
	temp = ''
	flag = True
	for w in code:
		if w in [' ', '\t']: flag = False
		else:
			if w == '\n':
				temp = ''
			output += temp
			temp = ''
			flag = True
		
		if flag: output += w
		else:temp += w
		
	return output
	
def get_test_data(file_name, data_list=[]):
	# read file and make test data
	test_data = {}
	a = read_file(file_name)

	# make index list
	idx_list = []
	for elem in data_list:
		idx = a.find(elem)
		idx_list.append(idx)
		
	idx_list.sort()
	
	# make dictionary
	m = len(idx_list)
	i = 0
	while i < m:
	
		if i+1 < m:
			idx = idx_list[i]
			colon = a.find(':', idx+1)
			name = a[idx:colon]
			st = colon+1
			next = idx_list[i+1]
			content = a[st:next]
			test_data[name] = content
		else:
			idx = idx_list[i]
			colon = a.find(':', idx+1)
			name = a[idx:colon]
			st = colon+1
		
			content = a[st:]
			test_data[name] = content
		
		i += 1
	# tab to space
	for name in test_data:
		test_data[name] = test_data[name].replace('\t','    ')
		
	# remove space after line
	for name in test_data:
		test_data[name] = remove_space_after_line(test_data[name])
	
	return test_data
	
def preprocessing(target='', detail=0):
	from preprocessing.main import test
	
	if target != '':
		test_input = read_file(target)
		flag = test(test_input, test_set=False, detail=detail)
	else:
		path = 'preprocessing/test_set/'
		
		tests = read_file(path+'test_list')
		test_list = get_test_list(tests)
		
		for test_name in test_list:
			test_data = get_test_data(path+test_name, ['test_input','test_output'])
			
			print "TEST:", test_name
			flag = test(test_data)
			if flag == False:
				return False
	return True
	
def test_vi2cu_translator(target='', detail=0):
	from vi2cu_translator.main import test
	if target != '':
		test_input = read_file(target)
		flag = test(test_input, test_set=False, detail=detail)
	else:
		path = 'vi2cu_translator/test_set/'

		tests = read_file(path+'test_list')
		test_list = get_test_list(tests)
		
		for test_name in test_list:
			test_data = get_test_data(path+test_name, ['test_input','test_output','dtype_dict','return_dtype'])
		
			print "TEST:", test_name
			flag = test(test_data)
			if flag == False:
				return False
					
	return True
	
def divide_line(target='',detail=0):
	from general.divide_line.divide_line import test
	path = 'general/divide_line/test_set/'
	
	tests = read_file(path+'test_list')
	test_list = get_test_list(tests)

	if target != '': test_list = [target]
	for test_name in test_list:
		test_data = get_test_data(path+test_name,['test_input','test_output'])
		
		print "TEST:", test_name
		flag = test(test_data, detail=detail)
		if flag == False:
			return False
	return True
	
def parse_main(target='', detail=0):
	from parse_main.main import test
	
	
	if target != '':
		test_input = read_file(target)
		flag = test(test_input, test_set=False, detail=detail)
	else:
		path = 'parse_main/test_set/'
		tests = read_file(path+'test_list')
		test_list = get_test_list(tests)
		
		for test_name in test_list:
			test_data = get_test_data(path+test_name, ['test_input','test_output'])
			
			print "TEST:", test_name
			flag = test(test_data, detail=detail)
			if flag == False:
				return False
			
	return True

def translator(target='', detail=0):
	from vivaldi_translator import test
	
	if target != '':
		test_input = read_file(target)
		test(test_input, test_set=False, detail=detail)
	else:
		path = 'test_set/'
		tests = read_file(path+'test_list')
		test_list = get_test_list(tests)
		
		for test_name in test_list:
			test_data = get_test_data(path+test_name, ['test_input','test_output','target'])
			
			print "TEST:", test_name
			flag = test(test_data, detail=detail)
			if flag == False:
				return False
	return True
	
def get_detail(argv):
	detail = 0

	detail_list = ['-D','-d','-detail']

	for symbol in detail_list:
		if symbol in argv:
			idx = argv.index(symbol)
			detail = int(argv[idx+1])
			del(argv[idx+1])
			del(argv[idx])
	
	return detail

if __name__ == "__main__":

	n = len(sys.argv)
	if n == 1:
		print "Usage: python run_test.py [options]"
		print "Options"
		print "  preprocessing		preprocessing before compiler"
		print "  vi2cu_translator	vivaldi to CUDA translation"
		print "  divide_line		test divide_line work"
		print "  parse_main		parse main test"
		print "  translator		test translator"
		
		exit()
		
	test_name = sys.argv[1]
	flag = False
	all_flag = False
	if test_name == 'all': # test all modules
		all_flag = True
	
	if test_name == 'divide_line' or all_flag:
		print "TEST DIVIDE LINE"
		print "========================================================"
		detail = get_detail(sys.argv)
		print "DETAIL", detail
		
		n = len(sys.argv)

		if n == 2:
			divide_line(detail=detail)
		elif n == 3:
			divide_line(sys.argv[2],detail=detail)
		
	if test_name == 'preprocessing' or all_flag:
		print "TEST PREPROCESSING"
		print "========================================================"
		detail = get_detail(sys.argv)
		
		print "DETAIL", detail
		n = len(sys.argv)
		if n == 2:
			preprocessing(detail=detail)
		elif n == 3:
			preprocessing(sys.argv[2], detail=detail)
		
	if test_name == 'vi2cu_translator' or all_flag:
		print "TEST VI2CU_TRANSLATOR"
		print "========================================================"
		detail = get_detail(sys.argv)
		
		print "DETAIL", detail
		n = len(sys.argv)
		if n == 2:
			test_vi2cu_translator(detail=detail)
		elif n == 3:
			test_vi2cu_translator(sys.argv[2], detail=detail)
		
	if test_name == 'parse_main' or all_flag:
		print "TEST PARSE_MAIN"
		print "========================================================"
		flag = preprocessing()
		if flag == False:
			print "must pass preprocessing test before parse_main"
			exit()
		
		detail = get_detail(sys.argv)

		print "DETAIL", detail
		n = len(sys.argv)
		if n == 2:
			parse_main(detail=detail)
		elif n == 3:
			parse_main(sys.argv[2],detail=detail)
			
		
	if test_name == 'translator' or all_flag:
		print "TEST TRANSLATOR"
		print "========================================================"
		detail = get_detail(sys.argv)

		print "DETAIL", detail
		
		n = len(sys.argv)
		if n == 2:
			translator(detail=detail)
		elif n == 3:
			translator(sys.argv[2],detail=detail)
		
		
