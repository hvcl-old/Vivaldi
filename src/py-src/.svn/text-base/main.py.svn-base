# import common module
import os
import sys
VIVALDI_PATH = os.environ.get('vivaldi_path')
sys.path.append(VIVALDI_PATH+'/src')
from common import *
#####################################################################

from mpi4py import MPI
import mpi4py
import multiprocessing
from Vivaldi_misc import *
from Vivaldi_functions import Vivaldi_functions
import subprocess

# match execid and compute unit type
computing_unit_list = {}

# mpi init
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


def make_mpi_spawn_info():
	# spawn children at once using Spawn_multiple
	MPI4PYPATH = os.path.abspath(os.path.dirname(mpi4py.__path__[0]))
	COMMAND = []
	ARGS = []
	MAXPROCS = []
	INFO = []

	# os.system("which nvprof")
	# os.system("nvprof --output-profile out.%p ")
	NVPROF_CMD = ["nvprof --output-profile out.%p "]
	# NVPROF_CMD = ["os.system("+"\"nvprof --output-profile out.%p\") "]
	# main manager
	##########################################################################
	filename = "Vivaldi_main_manager.py"
	unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
	info = MPI.Info.Create()
	
	COMMAND += [sys.executable]
	# COMMAND += NVPROF_CMD
	ARGS += [[unit, '-L', log_type,'-G','off','-B',VIVALDI_BLOCKING]]
	MAXPROCS += [1]
	INFO += [info]
	
	# memory manager
	##########################################################################
	filename = "Vivaldi_memory_manager.py"
	unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
	info = MPI.Info.Create()

	# COMMAND += NVPROF_CMD
	COMMAND += [sys.executable]
	ARGS += [[unit, '-L', log_type,'-B',VIVALDI_BLOCKING,'-S',VIVALDI_SCHEDULE,'-D',VIVALDI_DYNAMIC]]
	MAXPROCS += [1]
	INFO += [info]
		
	# scheduler
	##########################################################################
	filename = "Vivaldi_reader.py"
	unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
	info3 = MPI.Info.Create()

	# COMMAND += NVPROF_CMD
	COMMAND += [sys.executable]
	ARGS += [[unit, '-L', log_type,'-G','off','-B',VIVALDI_BLOCKING]]
	MAXPROCS += [1]
	INFO += [info3]
	
	# computing units
	############################################################################
	i = 4
	for computer in device_info:
		for device_type in device_info[computer]:
			if device_info[computer][device_type] == 0:continue
			filename = device_type+"_unit.py"
			unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
			info = MPI.Info.Create()
			if computer != name:
				hostfile = HOSTFILE_PATH + '/' + computer+'_hostfile'
				info.Set("hostfile", hostfile)

			COMMAND += [sys.executable]
			ARGS += [[unit, str(i), '-L', log_type,'-G',GPUDIRECT,'-B',VIVALDI_BLOCKING]]
			MAXPROCS += [device_info[computer][device_type]]
			INFO += [info]
			i = i + device_info[computer][device_type]


	return COMMAND, ARGS, MAXPROCS, INFO
	pass

# Vivaldi main manager related function
###############################################################################
def run_main_manager():
	comm.send(computing_unit_list,	dest=1, tag=1)
	global parsed_Vivaldi_functions
	comm.send(parsed_Vivaldi_functions,		dest=1,	tag=1)

# Vivaldi memory manager related function
###############################################################################
def run_memory_manager():
	comm.send(computing_unit_list, dest=2, tag=2)
	pass

# Vivaldi scheduler related function
###############################################################################
def run_reader():
	pass


# computing_unit disconnect function
################################################################################
def make_computing_unit_list():
	# 0 is main process
	# computing unit start from 1
	i = 4
	for computer in device_info:
		for device_type in device_info[computer]:
			for execid in range(device_info[computer][device_type]):
				computing_unit_list[i] = {'computer':computer,'device_type':device_type}
				i = i + 1
	pass

def clean_computing_unit():
	rank = comm.Get_rank()
	size = comm.Get_size()
	for i in range(size-1):
		tag = 5
		if i+1 == 1: tag = 1
		if i+1 == 2: tag = 2
		comm.send(rank,		dest=i+1,	tag=tag)
		comm.send("finish",	dest=i+1,	tag=tag)
	log("head waiting disconnection", 'general', log_type)
	comm.Barrier()
	comm.Disconnect()
	log("head finish", 'general', log_type)

def get_CPU_execid_list(computer=None):
	execid_list = []
	for elem in computing_unit_list:
		if computer == None or computer == computing_unit_list[elem]['computer']:
			if computing_unit_list[elem]['device_type'] == 'CPU':
				execid_list.append(elem)

	return execid_list

def get_GPU_execid_list(computer=None):
	execid_list = []
	for elem in computing_unit_list:
		if computer == None or computer == computing_unit_list[elem]['computer']:
			if computing_unit_list[elem]['device_type'] == 'GPU':
				execid_list.append(elem)

	return list(execid_list)

def deploy_CPU_function(dst=None, code=None):
	comm.isend(rank,			dest=dst,   tag=4)
	comm.isend("deploy_func",	dest=dst,	tag=4)
	comm.isend(code,			dest=dst,   tag=4)

def deploy_GPU_function(dst, code):
	comm.send(rank,				dest=dst,	tag=5)
	comm.send("deploy_func",	dest=dst,	tag=5)
	comm.send(code,				dest=dst,	tag=54)
	
# for easy programming
################################################################################
def divide_args(args):
	### <Summary> :: divide arguments and return list of args
	### Type of args :: str
	### Return type :: list
	args_list = []
	t = args
	last = 0
	bcnt = 0
	i = 0
	#remove blanket
	for s in args:
		if s == '(' or s == '[' or s == '{':bcnt = bcnt + 1
		if s == ')' or s == ']' or s == '}':bcnt = bcnt - 1
		if s == ',' and bcnt == 0:
			temp = args[last:i]
			if temp.isdigit(): temp = int(args[last:i])
			args_list.append(temp)
			last = i + 1
		i = i + 1
	if last < i:
		temp = args[last:i]
		if temp.isdigit(): temp = int(args[last:i])
		args_list.append(temp)
	return args_list

def strip_func(line):
	### <Summary> :: distinguish function name and function argument
	### Type of line :: str
	### if function not exist in this line
	### return False,"","",""
	### if function exist in this line
	### return True, func_name, func_args, after_func
	### ex) if(a<3):a=a+1
	### return True, if, a<3, :a=a+1
	tf = False
	f = line.find('(')
	if f != -1:
		name = line[:f]
		if name.strip().count(' ') == 0:pass
		else: return False, "", "", ""
		args = line[f+1: line.rfind(')')]
		after = line[line.rfind(')')+1:]
		return True, name, args, after
	return False, "", "", ""



if __name__ == "__main__":
	#parse argument
	if '-help' in sys.argv or '--help' in sys.argv or '-H' in sys.argv or len(sys.argv) == 1:
		print "Usage: Vivaldi [file] [options]"
		print "Options:"
		print "  -L"
		print "  -L time\t\t\tDisplay time log"
		print "  -L parsing\t\t\tDisplay parsing log"
		print "  -L general\t\t\tDisplay general log"
		print "  -L detail\t\t\tDisplay detail log"
		print "  -L image\t\t\tsave all intermediate results"
		print "  -L all\t\t\tDisplay every log"
		print "  -L progress\t\t\tDisplay which process is working with time progress"
		print "  -L retain_count\t\tDisplay change of retain_count during execution"
		print "  -hostfile" 
		print "  -hostfile hostfile_name \tinclude machine list in clusters"
		print "  -G"
		print "  -G on\t\t\t\tturn on GPU direct(default)"
		print "  -G off\t\t\tturn off GPU direct"
		print "  -B"
		print "  -B true\t\t\tblocking data transaction"
		print "  -B false\t\t\tnonblocking data trasnsaction(default), it will overlap data transaction with calculation"
		print "  -S\t\t\t\tscheduling algorithm"
		print "  -S round_robin\t\tround_robin scheduling"
		print "  -S locality\t\t\tlocality aware scheduling, minimize data transaction. (default)"
		print "  -D "
		print "  -D \t\t\t\tdynamic scheduling(default)"
		print "  -D \t\t\t\tstatic shceduling"
		exit()

	if '-template' in sys.argv:
		n = len(sys.argv)
		idx = sys.argv.index('-template')
		if idx+1 < n: name = sys.argv[idx+1]
		else: name = 'Vivaldi_template'

		VIVALDI_PATH = os.environ.get('vivaldi_path')
		template_path = VIVALDI_PATH+'/examples/Vivaldi_template.vvl'
		e = os.system('cp %s %s'%(template_path, name))
		if e != 0:
			print "linudx error code: ", e
			print "---------------------------------------------"
			print "VIVALDI ERROR: can not make Vivaldi template"
			print "Case1: Vivaldi path is not specified"
			print "Case2: Vivaldi_template.uVivaldi not exist in VIVALDI_PATH/examples/"
			print "Case3: linux cp operator failed"
			print "Case4: Vivaldi is out of date"
			print "if this is not one of four case, email to developer."
			print "contact, email: "
			print "\twoslakfh1@unist.ac.kr, master student"
			print "\twkjeong@unist.ac.kr, professor"
			print "------------------------------------"
		exit()

	if '-L' in sys.argv:
		idx = sys.argv.index('-L')
		log_type = sys.argv[idx+1]

	GPUDIRECT = "on"
	if '-G' in sys.argv:
		idx = sys.argv.index('-G')
		GPUDIRECT = sys.argv[idx+1]

	VIVALDI_BLOCKING = "False"
	if '-B' in sys.argv:
		idx = sys.argv.index('-B')
		VIVALDI_BLOCKING = sys.argv[idx+1]

	VIVALDI_SCHEDULE = 'local'
	if '-S' in sys.argv:
		idx = sys.argv.index('-S')
		VIVALDI_SCHEDULE = sys.argv[idx+1]

	VIVALDI_DYNAMIC = 'Dynamic'
	if '-D' in sys.argv:
		idx = sys.argv.index('-D')
		VIVALDI_DYNAMIC = sys.argv[idx+1]


	device_info = {}
	
	device_info[name] = {}
	VIVALDI_PATH = os.environ.get('vivaldi_path')
	HOSTFILE_PATH = VIVALDI_PATH+'/hostfile'
	hostfile = HOSTFILE_PATH+"/vivaldi_machinefile"

	machinefile = read_file(hostfile)
		
	
	my_file = ''
	flag = False

	for line in machinefile.split('\n'):
		line = line.strip()
		if line == '':continue
			
		if ':' in line:
			if line[:-1] == name: flag = True
			else: flag = False
			continue

		if flag:
			if '-CPU' in line:
				idx = line.find('=')
				num = line[idx+1:].strip()
				device_info[m_key]['CPU'] = int(num)
			elif '-GPU' in line:
				idx = line.find('=')
				num = line[idx+1:].strip()
				device_info[m_key]['GPU'] = int(num)
			elif '-G' in line:
				idx = line.find('-G')
				GPUDIRECT = line[idx+2:].strip()
			else:
				idx = line.find(' ')
				if idx != -1:
					m_key = line[:idx].strip()
					device_info[m_key] = {}
				my_file += line + '\n'

	machinefile = my_file
	for line in machinefile.split('\n'):
		if line.strip() == '':continue
		hostfile = line.split(' ')[0]
		f = open(HOSTFILE_PATH+'/'+hostfile+'_hostfile','w+')
		f.write(line)
		f.close()

	# make machine lists
	####################################################################################################
	if machinefile.strip() == '':
		print "HERERE?"
		device_info = {}
		n = 2
		try:
			import pycuda.driver as cuda
			import pycuda.autoinit
			print "number of GPU",cuda.Device.count()
			n = cuda.Device.count()
		except:
			print "NO PYCUDA"
		device_info[name] = {'CPU':0,'GPU':n}

#	if True:
	# ping and check, device is alive
	#################################################################################################
#	dead_list = []
#	for host in device_info:
#		if host == name: continue
#		ping = subprocess.Popen(
#				["ping", "-c", "4", host],
#				stdout = subprocess.PIPE,
#				stderr = subprocess.PIPE
#				)

#		out, error = ping.communicate()

#		temp = None
#		for line in out.split('\n'):
#			if 'packets transmitted' in line:
#				print line
#				temp = line.split(',')
#				break
#		total = temp[0].strip().split(' ')[0]
#		transmitted = temp[1].strip().split(' ')[0]
#		if total == transmitted: print "alive host: ", host
#		else:
#			print "dead host: ",host
#			dead_list.append(host)

#	for host in dead_list:
#		del(device_info[host])
	
	
#	print "ALIVE HOST", device_info.keys()
	
	
#	for host in device_info:
#		filename = "counter.py"
#		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
#		info = MPI.Info.Create()

#		COMMAND = [sys.executable]
#		ARGS = [[unit]]
#		MAXPROCS = [1]
#		INFO = [info]

#		child = MPI.COMM_SELF.Spawn_multiple(COMMAND, ARGS, MAXPROCS,info=INFO, root=0)

#		CPU_count = child.recv(source=0, tag=1)
#		GPU_count = child.recv(source=0, tag=2)

		# for only GPU test, CPU count is setted 0
#		CPU_count = 0
#		device_info[host]['CPU'] = CPU_count
#		device_info[host]['GPU'] = GPU_count
	
#		print "CPU_count", CPU_count
#		print "GPU_count", GPU_count
#		child.Disconnect()
#		pass
	
	#log("rank%d, main launch at %s"%(rank, name), 'general', log_type)
	

	# computing unit manager
	#########################################################################################
	make_computing_unit_list()	
	COMMAND, ARGS, MAXPROCS, INFO = make_mpi_spawn_info()
	os.system('# Open MPI						\
		if [ ! -z ${OMPI_COMM_WORLD_RANK} ] ; then	\
			echo ${OMPI_COMM_WORLD_RANK}			\
			rank=${OMPI_COMM_WORLD_RANK}			\
			echo $rank								\
		fi')			
	# NEW_COMMAND = ['# Open MPI						\
		# if [ ! -z ${OMPI_COMM_WORLD_RANK} ] ; then	\
			# echo ${OMPI_COMM_WORLD_RANK}			\
			# rank=${OMPI_COMM_WORLD_RANK}			\
			# echo $rank								\
		# fi'											
	# ]
	NEW_COMMAND = []
	index=0
	for elem in COMMAND:
		# if index in [3,4]:
			# # NEW_COMMAND.append(["CUDA_PROFILE=1 ", elem])
			# # NEW_COMMAND.append(elem + " CUDA_PROFILE=1")
			# # NEW_COMMAND.append("/cm/shared/apps/cuda55/toolkit/5.5.22/bin/nvprof  " + elem + " ")
			# NEW_COMMAND.append("/cm/shared/apps/cuda55/toolkit/5.5.22/bin/nvprof --output-profile out.%p " + elem + " ")
			# # NEW_COMMAND.append("nvprof --output-profile out.%p " + elem)
			# # NEW_COMMAND.append(elem)
		# else:
			# NEW_COMMAND.append(elem)
		# index += 1
		# NEW_COMMAND.append("/cm/shared/apps/cuda55/toolkit/5.5.22/bin/nvprof  --output-profile out.%p " + elem)
		# NEW_COMMAND.append("/cm/shared/apps/cuda55/toolkit/5.5.22/bin/nvprof " + elem)
		NEW_COMMAND.append(elem)
	COMMAND = NEW_COMMAND
	
	#!!! Quan: Call nvprof here
	# os.system("nvprof --output-profile out.%p ")
	#os.system("module load cuda55/profiler")
	#os.system("which nvprof")
	# os.system("nvprof --output-profile out.%p ")
	# for elem in COMMAND:
		# print elem
	
	####
	computing_unit_comm = MPI.COMM_SELF.Spawn_multiple(
			COMMAND, ARGS, MAXPROCS,
			info=INFO, root=0)
	comm = computing_unit_comm.Merge()

	# Read Vivaldi code from file	
	###########################################################################################
	global parsed_Vivaldi_functions	
	log("rank%d, Argument List: %s"%(rank,str(sys.argv)), 'detail', log_type)
	filename = sys.argv[1]
	Vivaldi_code = read_file(filename)
	parsed_Vivaldi_functions = Vivaldi_functions(Vivaldi_code, log_type)

	# device check, send gpu_direct list
	execid_GPU_list = get_GPU_execid_list()
	for dest in execid_GPU_list:
		live = comm.recv(source=dest, tag=777)
		if live == "device_unavailable":
			del(computing_unit_list[dest])
	execid_GPU_list = get_GPU_execid_list()

	# check and deploy non-parallel Vivaldi and python code
	############################################################################################

	execid_GPU_list = get_GPU_execid_list()

	cuda_code = parsed_Vivaldi_functions.get_cuda_code()
	attachment = parsed_Vivaldi_functions.get_attachment()

	full_GPU_code = attachment + cuda_code

	log(full_GPU_code, 'parsing', log_type)
	for i in execid_GPU_list:
		deploy_GPU_function(i, full_GPU_code)
	
	#a = parsed_Vivaldi_functions.get_python_code('main') + '\n' + 'main()'
	
	# Vivaldi main manager
	##########################################################################################
	main_manager_rank = 1
	run_main_manager()
	
	# Vivaldi memory manager
	##########################################################################################
	memory_manager_rank = 2
	run_memory_manager()

	# Vivaldi reader
	############################################################################################
	reader_rank = 3
	run_reader()

	recv = comm.recv(source=main_manager_rank,tag=9)
	log("rank%d, main recv %s"%(rank,str(recv)), 'general', log_type)
	clean_computing_unit()
	
	os.system('rm a.out -f')
	os.system('rm asdf.cu -f')
	os.system('rm tmp_compile.cu -f')
	
