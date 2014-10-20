import os, sys
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()


# spawn child
def scheduler_update_computing_unit(cud):
	dest = 1
	tag = 5
	comm.isend(rank,					   dest=dest,	 tag=tag)
	comm.isend("update_computing_unit",	   dest=dest,	 tag=tag)
	comm.isend(cud,						   dest=dest,	 tag=tag)
def spawn_all():
	import mpi4py
	# local variables
	MPI4PYPATH = os.path.abspath(os.path.dirname(mpi4py.__path__[0]))
	COMMAND = []
	ARGS = []
	MAXPROCS = []
	INFO = []
	
	global device_info
	device_info = {}
	m_key = ''
	# local_functions
	def read_hostfile():
		# "read hostfile"
		global hostfile
		try:
			f = open(hostfile)
			x = f.read()
			f.close()
		except:
			print "Vivaldi warnning"
			print "=============================="
			print "hostfile not found"
			print ""
			print "hostfile example"
			print "nodename1"
			print "-GPU=2"
			print "nodename2"
			print "-GPU=2"
			print "=============================="
			exit()
		return x
	def set_device_info(hostfile):
		my_file = ''

		for line in hostfile.split('\n'):
			line = line.strip()
			if line == '':continue
					
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
				m_key = line
				device_info[m_key] = {}				
	def add_main_unit(COMMAND, ARGS, MAXPROCS, INFO):
		# starter
		##########################################################################
		filename = "main_unit.py"
		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
		info = MPI.Info.Create()
		info.Set("host", name)
		COMMAND += [sys.executable]
		ARGS += [[unit]]
		MAXPROCS += [1]
		INFO += [info]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_reader(COMMAND, ARGS, MAXPROCS, INFO):
		# scheduler
		##########################################################################
		filename = "Vivaldi_reader.py"
		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
		info3 = MPI.Info.Create()

		COMMAND += [sys.executable]
		ARGS += [[unit]]
		MAXPROCS += [1]
		INFO += [info3]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_scheduler(COMMAND, ARGS, MAXPROCS, INFO):
		# memory manager
		##########################################################################
		filename = "Vivaldi_memory_manager.py"
		unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
		info = MPI.Info.Create()
		
		COMMAND += [sys.executable]
		ARGS += [[unit]]
		MAXPROCS += [1]
		INFO += [info]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_CPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile):
		device_type = 'CPU'
		for host in device_info:
			if device_type not in device_info[host]:continue
			if device_info[host][device_type] == 0:continue
			filename = 'CPU_unit.py'
			unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
			info = MPI.Info.Create()
			info.Set("host", host)
			
			COMMAND += [sys.executable]
			ARGS += [[unit]]
			MAXPROCS += [device_info[host][device_type]]
			INFO += [info]
		
		return COMMAND, ARGS, MAXPROCS, INFO
	def add_GPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile):
		device_type = 'GPU'
		i = 3
		for host in device_info:
			if device_type not in device_info[host]:continue
			if device_info[host][device_type] == 0:continue
			filename = 'GPU_unit.py'
			unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
			info = MPI.Info.Create()
			info.Set("host", host)
			
			COMMAND += [sys.executable]
			ARGS += [[unit, str(i)]]
			MAXPROCS += [device_info[host][device_type]]
			INFO += [info]
			i = i + device_info[host][device_type]
			
		return COMMAND, ARGS, MAXPROCS, INFO
	def set_computing_unit_list():
		# computing unit start from 3
		# 0: main
		# 1: scheduler
		# 2: reader
		# n-1: main_unit
		
		computing_unit_list = {}
		rank = 3
		for host in device_info:
			for device_type in device_info[host]:
				for execid in range(device_info[host][device_type]):
					computing_unit_list[rank] = {'host':host,'device_type':device_type}
					rank = rank + 1	
					
		return computing_unit_list
		
	hostfile = read_hostfile()
	set_device_info(hostfile)
	COMMAND, ARGS, MAXPROCS, INFO = add_scheduler(COMMAND, ARGS, MAXPROCS, INFO)
	COMMAND, ARGS, MAXPROCS, INFO = add_reader(COMMAND, ARGS, MAXPROCS, INFO)
	COMMAND, ARGS, MAXPROCS, INFO = add_CPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile)
	COMMAND, ARGS, MAXPROCS, INFO = add_GPU(COMMAND, ARGS, MAXPROCS, INFO, hostfile)
	COMMAND, ARGS, MAXPROCS, INFO = add_main_unit(COMMAND, ARGS, MAXPROCS, INFO)
	new_comm = MPI.COMM_SELF.Spawn_multiple(
			COMMAND, ARGS, MAXPROCS,
			info=INFO, root=0)
	global comm
	comm = new_comm.Merge()
	
	global main_unit_rank
	main_unit_rank = comm.Get_size()-1
	computing_unit_list = set_computing_unit_list()
	scheduler_update_computing_unit(computing_unit_list)

# data management
def free_volumes(): # old function
	# old version function
	for data_name in data_package_list.keys():
		u = data_list[data_name]
		if u in retain_list:
			for elem in retain_list[u]:
				mem_release(elem)
			del(retain_list[u])

		del(data_package_list[data_name])

try:
	cache
	print "Vivaldi_init not initialize variables"
except:
	hostfile = 'hostfile'
	main_unit_rank = -1
		
# argument parsing
def Vivaldi_input_argument_parsing(argument_list):
	def hostfile_parsing():
		if '--hostfile' in sys.argv:
			idx = sys.argv.index('--hostfile')
			global hostfile
			hostfile = sys.argv[idx+1]
	hostfile_parsing()
	spawn_all()
	def option_parsing():
		if '-L' in argument_list:
			idx = argument_list.index('-L')
			#GPUDIRECT = argument_list[idx+1]
			#log_on()
	option_parsing()
	def get_filename(argument_list):
		filename = None
		skip = False
		i = 0
		for elem in argument_list[1:]:
			i += 1
			if skip:
				skip = False
				continue
			if elem.startswith('-'):
				skip = True
			else:
				filename = argument_list[i]		
		return filename
	filename = get_filename(argument_list)
	return filename
filename = Vivaldi_input_argument_parsing(sys.argv)

if filename != None:
	comm.send(rank,      dest=main_unit_rank, tag=5)
	comm.send("run",     dest=main_unit_rank, tag=5)
	comm.send(filename,  dest=main_unit_rank, tag=5)
else:
# interactive mode functions
	def interactive_mode():
		# interactive mode 
		################################################################################################
		import os
		import sys
		from code import InteractiveConsole
		from tempfile import mkstemp

		EDITOR = os.environ.get('EDITOR', 'vi')
		EDIT_CMD = '\e'

		class VivaldiInteractiveConsole(InteractiveConsole):
		#	def __init__(self, *args, **kwargs):
		#		self.last_buffer = [] # This holds the last executed statement
		#		InteractiveConsole.__init__(self, *args, **kwargs)
			"""
			def runsource(self, source, *args):
				from Vivaldi_translator_layer import line_translator
			#	source = line_translator(source, data_package_list)
				return InteractiveConsole.runsource(self, source, *args)
			"""
			
			def runsource(self, source, filename="<input>", symbol="single"):
			
				if source == '':
					return False
				if source == 'exit()':
					exit()
			
				try:
					code = self.compile(source, filename, symbol)
					eval_expression = True
				except (OverflowError, SyntaxError, ValueError):
					# Case 1
					self.showsyntaxerror(filename)
					return False
				
				if code is None:
					# Case 2
					return True
				
				global main_unit_rank
				
				comm.send(rank,      dest=main_unit_rank, tag=5)
				comm.send("runcode", dest=main_unit_rank, tag=5)
				comm.send(source,    dest=main_unit_rank, tag=5)
				
				x = comm.recv(       source=main_unit_rank, tag=5)
				
				# print output
				if len(x) == 0:
					#print, 
					pass
				else:
					print x.rstrip()
				
				return False
				
		#	def raw_input(self, *args):
		#		line = InteractiveConsole.raw_input(self, *args)
		#		return line
		
		def Vivaldi_interactive(_globals, _locals):
			"""
			Opens interactive console with current execution state.
			Call it with: `console.open(globals(), locals())`
			"""
			import readline
			import rlcompleter

			context = _globals
			context.update(_locals)
			readline.set_completer(rlcompleter.Completer(context).complete)
			shell = VivaldiInteractiveConsole(context)
			shell.interact()

		Vivaldi_interactive(globals(), locals())

	interactive_mode()

x = comm.recv(source=main_unit_rank, tag=5)