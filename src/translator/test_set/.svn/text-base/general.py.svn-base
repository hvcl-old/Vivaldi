# check this element is operator or not



operators = ['.','+','-','!','~','*','&','*','/','%','<<','>>','<','<=','>','>=','==','!=','^','|','&&','||','=','+=','-=','*=','/=','%=','<<=','>>=','&=','^=','|=']
	
def is_operator(elem):
	
	s_elem = elem.strip()

	if s_elem in operators:
		return True
	return False
	
	
# get indent of the input line
def get_indent(line): 
	s_line = line.strip()
	i_idx = line.find(s_line)
	indent = line[:i_idx]

	return indent
	
# find difference of two string
def dif(a, b):
	n = len(a)
	m = len(b)
	if n > m: n = m
	i = 0
	while i < n:
		if a[i] != b[i]:
			print "DIF"
			print "A"
			print a[i:]
			print "B"
			print b[i:]
			return False
		i += 1
	return True