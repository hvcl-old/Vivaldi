# preprocessing function for compiler is come here


# remove backslash like new line symbols

# remove comment in the code

# remove semicolon, it's just well known mistake

# stretch code, one line if, for, while is hard to compiler
	
def remove_line_break(code):
	i = 0 
	n = len(code)
	output = ''
	while i < n:
		w = code[i]
		# endline symbol
		if w == '\n':
			# tuple, dictionary, list and function 
			p_list = [('(',')'),('{','}'),('[',']')]
			flag = False
			for parenthesis in p_list:
				p_st = parenthesis[0]
				p_ed = parenthesis[1]
				
				if code[:i].count(p_st) != code[:i].count(p_ed):
					flag = True
					break
					
			# check VIVALDI SPECIAL CASE, modifier is multiple line
			# func().modifier()
			#       .modifier()
		
			# left striped next line start with dot
			if i+1 < n:
				ls_code = code[i+1:].lstrip()
				if len(ls_code) > 1 and ls_code[0] == '.':
					# remove end line
					# and skip space for looks good
					flag = True
					while True:
						if i+1 < n and code[i+1] in [' ', '\t']:
							i += 1
						else:
							break
					
		
			# if in the parenthesis
			if flag:
				# remove end line symbol
				i += 1
				# continue instead 'output += w' because there can be muliple endline symbol
				continue
				
		# line glue symbol backslash, '\\'
		if w == '\\':
			i += 2
			continue
			
		output += w
		i += 1
	return output
			
def remove_comment(code, flag='#'):
	output = ''
	flag = True
	for w in code:
		if w == '#': flag = False
		if w == '\n': flag = True
		if flag: output += w
	return output
	
def remove_semicolon(code):
	n = len(code)
	output = ''
	i = 0
	for w in code:
		if w == ';':
			if i + 1 >=  n:
				if code[i+1] == '\n':
					continue
			else:
				continue
		output += w
		i += 1
	
	return output
	
def stretch(code):
	# stretch one line if and for statement to multiple line
	# stretch based on symbol colon. but colon also used for python array
	# but currently I will assume, colon is not used for array index 

	n = len(code)
	output = ''
	indent = ''
	i_flag = True
	s_flag = False
	s_symbol = ''
	i = 0
	while i < n:
		w = code[i]
		
		# output buffer
		output += w
		# indent 
		if w in [' ','\t']:
			if i_flag: indent += w
		else: 
			i_flag = False
		
		if w == '\n': 
			i_flag = True
			indent = ''
		
		# skip if it's in the string
		if s_flag == False:
			if w in ['"', "'"]:
				s_symbol = w
				s_flag = True
		else:
			if w == s_symbol and code[i-1] != '\\':
				s_flag = False
				s_symbol = ''
		
		# find colon
		if w == ':' and s_flag == False and i+1 < n and code[i+1] != '\n':
			# 2014-04-2?
			# check what is this colon mean
			# there are two usage of colon
			# one is new block
			# another is dictionary or list
			
			# dictionary case
			p_list = [('{','}'),('[',']')]
			flag = True
			for parenthesis in p_list:
				p_st = parenthesis[0]
				p_ed = parenthesis[1]
				
				if code[:i].count(p_st) != code[:i].count(p_ed):
					flag = False
					break
					
			# VIVALDI modifier case
			if flag: 
				p_list = [('(',')')]
				for parenthesis in p_list:
					p_st = parenthesis[0]
					p_ed = parenthesis[1]
					
					if code[:i].count(p_st) != code[:i].count(p_ed):
						flag = False
						break
			
			# new block case
			if flag:
				# this is new block			
				# add end line symbol and indent
				output += '\n' + indent + '    '
			
				# remove unnecessary space 
				i += 1
				while True:
					if code[i] not in [' ','\t']:
						break
					i += 1
				continue
		i += 1
		
	return output
	
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
		
# doing preprocessing for incoming code
def preprocessing(code):
	#	tab to space
	#
	#######################################################
	code = code.replace('\t','    ')

	# Remove comment
	#
	####################################################### 
	code = remove_comment(code)

	# Remove line break
	#
	#######################################################
	# merge multiple line to one line
	code = remove_line_break(code)

	# Remove space after line
	#
	#######################################################
	# remove space and tab after line
	code = remove_space_after_line(code)
	
	if False: # not necessary, Python don't care semicolon
		# Remove semicolon
		#
		#######################################################
		# remove semicolon, it's frequent mistake
		# remove semicolon after remove space, because it's easier
			code = remove_semicolon(code)
	
	# Remove stretch
	#
	#######################################################
	# stretch one line if statement to multiple line
	# because it's easier to translate
	code = stretch(code)

	return code

def test(test_data, test_set=True, detail=0):

	if test_set:
		test_input = test_data['test_input']
		test_output = test_data['test_output']

		result = preprocessing(test_input)

		if str(result)== test_output:
			print "OK"
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
		result = preprocessing(test_input)
		print "PREPROCESSING"
		print "========================================"
		print result
		return False
