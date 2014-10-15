# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
path = VIVALDI_PATH+'/src'
if path not in sys.path:
	sys.path.append(path)
from common import *
#####################################################################

from mpi4py import MPI
f = open(VIVALDI_PATH+'/hostfile/vivaldi_machinefile')
machinefile = f.read()
f.close()

#print machinefile

my_file = ''
flag = False
name = MPI.Get_processor_name()
temp = ''
w_flag = False
for line in machinefile.split('\n'):
	line = line.strip()
	if line == '':continue
	if ':' in line:
		if line[:-1] == name: flag = True
		else: flag = False
		continue
		
	if flag:
		if line[0] == '-':
			pass
		else:
			if w_flag:
				my_file += temp
				temp = ''
				w_flag = False

			temp += line.strip() + '\n'

		if '=' in line:
			idx = line.find('=')
			num = int(line[idx+1:].strip())
			if num > 0: w_flag = True
	
if w_flag:
	my_file += temp
	temp = ''
	w_flag = False
machinefile = my_file

f = open(VIVALDI_PATH+'/hostfile/hostfile','w')
f.write(my_file)
f.close()
