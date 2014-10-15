import sys
from misc import read_file


def remove_first_endline(line):
	f_endl = line.find('\n')
	flag = True
	for i in range(f_endl+1):
		if line[i] not in [' ', '\t', '\n']:
			flag = False
	
	if flag:
		line = line[f_endl+1:]

	return line

def get_test(file_name):
	test = {}
	a = read_file(file_name)

	fin = a.find('test_input:')
	fout = a.find('test_output:')

	test['input'] = a[fin+len('test_input:'):fout]
	test['output'] = a[fout+len('test_output:'):]

	# remove first new line
	test['input'] = remove_first_endline(test['input'])
	test['output'] = remove_first_endline(test['output'])
	
	return test
	
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

def divide_line():
	from functions.divide_line.divide_line import test
	path = 'functions/divide_line/test_set/'
	
	tests = read_file(path+'test_list')
	test_list = tests.split('\n')

	for test_name in test_list:
		if test_name == '':continue
		test_data = get_test(path+test_name)

		print "TEST:", test_name
		test(test_data['input'], test_data['output'])
		
	return False
	
def split_into_block_and_code(target=''):
	from functions.split_into_block_and_code.split_into_block_and_code import test
	path = 'functions/split_into_block_and_code/test_set/'

	tests = read_file(path+'test_list')
	test_list = tests.split('\n')

	if target != '': test_list = [target]
	for test_name in test_list:
		if len(test_name) > 0 and test_name[0] == '#':continue
		if test_name == '':continue
		test_data = get_test_data(path+test_name, ['test_input','test_output'])

		print "TEST:", test_name
		test(test_data)
	return True
	
def code_to_line_list():
	from functions.code_to_line_list.code_to_line_list import test
	path = 'functions/code_to_line_list/test_set/'

	tests = read_file(path+'test_list')
	test_list = tests.split('\n')
	
	for test_name in test_list:
		if test_name == '':continue
		test_data = get_test(path+test_name)

		test_output = test_data['output'].split('\n')
		
		print "TEST:", test_name
		test(test_data['input'], test_output)
	return False
	
if __name__ == "__main__":
	
	n = len(sys.argv)
	if n == 1:
		print "Usage: python run_test.py [options]"
		print "Options"
		print "  divide_line			divide a line to multiple elements"
		print "  split_into_block_and_code	split a block into small blocks and else"
		print "  code_to_line_list	"
		exit()
	
	test_name = sys.argv[1]
	flag = False
	if test_name == 'divide_line':
		divide_line()
		
	elif test_name == 'split_into_block_and_code':
		if n == 2:
			split_into_block_and_code()
		elif n == 3:
			split_into_block_and_code(sys.argv[2])
		
	elif test_name == 'code_to_line_list':
		code_to_line_list()