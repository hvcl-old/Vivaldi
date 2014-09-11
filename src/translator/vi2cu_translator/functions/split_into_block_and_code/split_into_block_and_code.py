# a block is consist of many small blocks and codes
# split these and make a list

import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
sys.path.append(VIVALDI_PATH + "/src/translator")

from common_in_translator import *
from general.divide_line.divide_line import divide_line

def compare_indent(pre, new):
	# replace tab to four space
	c_pre = pre.replace('\t','    ')
	c_new = new.replace('\t','    ')
	if c_pre == c_new:
		return True

	return False

def split_into_block_and_code(in_block):
	# initialization
	###################################################
	block_list = []
	code_list = []

	line_list = in_block.split('\n')

	code = ''
	block = ''

	flag = False # True is block, False is code

	# there are two important thing
	# find block start, block always start with colon
	# find block end, block ends determined with indention
	
	indent = 0
	for line in line_list:
		l_line = line.strip()
		if l_line != '':
			indent = get_indent(line)
			break
	
	# implementation
	###################################################
	if line_list[-1] == '':
		line_list = line_list[:-1]
	
	for line in line_list:
		# recover end line symbol
		if True:
			idx = in_block.find(line)
			n = len(line)
			if idx+n < len(in_block):
				if in_block[idx+n] == '\n':
					line += '\n'
					
			if line == '':
				line = '\n'
	
		# check indent
		new_indent = get_indent(line)
		if False:
			# debug
			print "line", line, len(indent), len(new_indent)
		
		# there are n case block finish
		# 1. something in -4 indent
		#    but need exception handling with block indent 4
		# 2. if indent is 4 than zero indent and string start with not space finish block 
		#
		if flag:
			i_flag = False
			# case 1
			if len(indent) == len(new_indent) and new_indent != '': i_flag = True
			# case 2
			if new_indent == '' and len(line) > 0 and line[0] not in [' ','\t','\n']: 
				i_flag = True
			if i_flag:
				# this line is code, because indent is same
				block_list.append(block)
				block = ''
				flag = False
				
		# colon in the line.
		# it's start of new block
		# because CUDA do not have grammar like a[:3]
		f_colon = line.find(':')
		if f_colon == -1:
			# there are no colon
			if flag:
				block += line
			else:
				code += line
		else:
			# there are colon
			# if currently not in the block

			if flag == False:
				# block starts with if, elif, else, for, while
				l_line = line.strip()
				
				g_list = ['if','elif','else','while']
				g_flag = False
				for grammar in g_list:
					if l_line.startswith(grammar):
						g_flag = True
						break
				
				if g_flag:
					# attach this to code
					code += line

				elif l_line.startswith('for'):
					# for is special case
					# because local definition is included in the for statement
					block += line
					
				code_list.append(code)
				code = ''
				flag = True
			else:
				block += line

	if code is not '':
		code_list.append(code)
		code = ''

	if block is not '':
		block_list.append(block)
		block = ''
		
	if False:
		# debug
		print "block_list"
		for elem in block_list:
			print "#block", elem

		print "code_list"
		for elem in code_list:
			print "#code", elem
		assert(False)
	return block_list, code_list

#def test(test_input, test_block_list, test_code_list):
def test(test_data):
	test_input = test_data['test_input']
	test_output = test_data['test_output']

	block_list, code_list = split_into_block_and_code(test_input)
	flag = True
	output = '\n'
	for code in code_list:
		output += '#code\n'
		output += code
	for block in block_list:
		output += '#block\n'
		output += block

	flag = dif(output, test_output)
	
	if flag:
		print "OK"
	#	print output
		return True
	else:
		print "FAILED"
		print "test_input:", test_input
		print "test_output:", test_output
		print "result:", output
		return False