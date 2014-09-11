# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
VIVALDI_PATH = '/home/hschoi/Vivaldi'
path = VIVALDI_PATH+'/src/translator'
if path not in sys.path:
	sys.path.append(path)

from common_in_translator import *
#####################################################################

# split line to variables and operators
# example) a + b is divide to ['a', '+', 'b']

def check_OPERATORS(temp, elem_list):
	flag = False
	merge_able = ['+','-','*','/','=','!','>','<']
	for operator in OPERATORS:
		if temp.endswith(operator):
			m = len(operator)
			operand = temp[:-m].strip()
			
			m_flag = False
			if operator == '=':
				if operand == '':
					m_flag = True
					if len(elem_list) == 0: elem_list.append(operator)
					else:
						if elem_list[-1][-1] in merge_able: 
							elem_list[-1] += operator
						else:
							m_flag = False
				
			if m_flag == False:
				if operand.strip() != '':
					elem_list.append(operand.strip())
				if operator.strip() != '':
					elem_list.append(operator.strip())
					
			flag = True
			break
	return flag
		
def check_WORD_OPERATORS(temp, elem_list, next):
	
	flag = False
	for operator in WORD_OPERATORS:
		if temp.endswith(operator):
			m = len(operator)
			operand = temp[:-m].strip()
	
			previous = ''
			if len(temp) > m:
				previous = temp[-m-1]
				
			if previous not in [' ', '']: continue
			if next not in [' ', '']: continue
			
			if operand.strip() != '':
				elem_list.append(operand.strip())
			if operator.strip() != '':
				elem_list.append(operator.strip())
			
			flag = True
	return flag
		
def divide_line(line):
	elem_list = []
	i = 0
	line = line.strip()
	n = len(line)
	temp = ''
	
	while i < n:
		w = line[i]
			
		# test temp
		flag = check_OPERATORS(temp, elem_list)
		if flag == False: flag = check_WORD_OPERATORS(temp, elem_list, w)
		if flag: temp = ''
				
		start = i
		flag1, i = skip_string(line, i)
		if flag1:
			temp += line[start:i]
		
		if i >= n: break
		start = i
		flag2, i = skip_parenthesis(line, i)
		if flag2:
			if temp.strip() != '':
				elem_list.append(temp.strip())
				temp = ''
			temp += line[start:i]
			
		if flag1 == False and flag2 == False:
			temp += w
			i += 1
	
	if temp.strip() != '':
		flag = check_OPERATORS(temp, elem_list)
		if flag == False: flag = check_WORD_OPERATORS(temp, elem_list, next)
		
		if flag == False:
			elem_list.append(temp.strip())
	
	return elem_list

def test(test_data,detail=0):
	test_input = test_data['test_input']
	test_output = test_data['test_output']

	result = divide_line(test_input)

	if str(result).strip() == test_output.strip():
		print "OK"
		if detail > 0: print result
		return True
	else:
		print "FAILED"
		print "test_input:", test_input
		print "test_output:", test_output
		print "result:", result
		print "\n"
		return False

def test_string(input):
	return divide_line(input)

def import_test():
	print "Hello"
	assert(False)
