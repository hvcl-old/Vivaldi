from mpi4py import MPI
from Vivaldi_misc import *
from Vivaldi_load import *
from Vivaldi_memory_packages import *


# mpi init
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
	
# initialize variables
main_unit_rank = size-1

working_list = {}

data_list = [] #which computing unit have data
source_list_dict = {}
data_packages = {}

remaining_write_count_list = {}
valid_list = {}

retain_count = {}
dirty_dict = {}

computing_unit_dict = {}
computing_unit_list = []
processor_list = []
idle_list = []
work_list = {}
reserved_dict = {}

# work list will divide to kernel running and data copy
kerner_list = {}
copy_list = {}

making_data = {}

compositing_method = {}

function_list = []
memcpy_tasks = {}
memcpy_tasks_to_harddisk = {}

synchronize_flag = False

VIVALDI_BLOCKING = False
VIVALDI_SCHEDULE = False
VIVALDI_DYNAMIC = True

log_type = False


# input split
ou_id = 0
depth_dict = {}

# MPI communication
def synchronize():
	global synchronize_flag
	if False: # Debug 
		print synchronize_flag
		print function_list
		print memcpy_tasks
		print work_list
		print data_list
		print memcpy_tasks_to_harddisk
		print working_list
		print "DDDDD", idle_list
	
	if synchronize_flag != True: return False
	if function_list != []: 
#		print "sync, function_list:", function_list
		return False
	if memcpy_tasks != {}: 
#		print "sync, memcpy_task:", memcpy_task
		return False
	if work_list != {}: 
#		print "sync, work_list:", work_list
		return False
	if data_list != []: 
#		print "sync, data_list:", data_list
		return False
	if memcpy_tasks_to_harddisk != {}:
#		print "sync, memcpy_task_to_harddisk:", memcpy_tasks_to_harddisk
		return False
	if working_list != {}:
#		print "sync, working_list:", working_list
		return False
	
#	print "SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS"
	for dest in idle_list:
		comm.isend(rank,			dest=dest,	tag=5)
		comm.isend("synchronize",	dest=dest,	tag=5)
		comm.recv(source=dest, tag=999)
	
	comm.isend(True, dest=size-1, tag=999)
	synchronize_flag = False
	
def disconnect():
	print "Disconnect", rank, name
	comm.Disconnect()
	MPI.Finalize()
	exit()
def memcpy_p2p(source, dest, task):
	comm.isend(rank,             	dest=source,		tag=5)
	comm.isend("memcpy_p2p_send",	dest=source,		tag=5)
	comm.isend(dest,				dest=source,		tag=56)
	comm.isend(task,				dest=source,		tag=56)
	
	if dest not in working_list:
		working_list[dest] = []
	working_list[dest].append(task.dest)
#	print "COPY", source, dest, working_list
	
# scheduling
def temp_func1(for_save=False, source_list=None): # memcpy
	return_flag = True

	if for_save: cur_dict = memcpy_tasks_to_harddisk
	else: cur_dict = memcpy_tasks

	source_list = [elem for elem in source_list if elem in idle_list]
	if source_list == []:return False

	if 2 in source_list:
		source_list.remove(2)
		source_list.append(2)

	for u in cur_dict.keys():
		if u not in valid_list:continue
		for ss in cur_dict[u].keys():
			if ss not in valid_list[u]:continue
			for sp in cur_dict[u][ss].keys():
				if sp not in valid_list[u][ss]:continue
				for task in list(cur_dict[u][ss][sp]):

					tf = time.time()

					if task.start != None:
						source = task.start
						if source not in idle_list: continue
						if source not in valid_list[u][ss][sp]: continue

						dest = task.execid
						if VIVALDI_BLOCKING and dest not in idle_list:
							continue

						flag = 1
						if dest == source: flag = 2
						if dest == 1: flag = 2

						memcpy_p2p(source, dest, task)
						cur_dict[u][ss][sp].remove(task)
						if cur_dict[u][ss][sp] == []: del cur_dict[u][ss][sp]
						if cur_dict[u][ss] == {}: del cur_dict[u][ss]
						if cur_dict[u] == {}: del cur_dict[u]
						
						
#						print "Copy", source, dest, time.time()
						log("Copy %d to %d"%(source, dest),'progress', log_type)
	
						"""
						idle_list.remove(source)
						work_list[source] = []
						if source in source_list:
							source_list.remove(source)
	
						if VIVALDI_BLOCKING and flag == 1:
							idle_list.remove(dest)
							if dest in source_list:
								source_list.remove(dest)
							work_list[dest] = []
						
						if source_list == []:
							flag_times["time1"] += time.time() - st
							return 
						"""
					elif task.start == None:
						cur_list = valid_list[u][ss][sp] # valid value
						f_source_list = [elem for elem in source_list if elem in cur_list]
	
						dest = task.execid
	
						if VIVALDI_BLOCKING and dest not in idle_list:
							continue
	
						for source in f_source_list:
							flag = 1
							if dest == source: flag = 2
							if dest == 1: flag = 2

#							log("Copy %d to %d"%(source, dest),'progress', log_type)
#							print "Copy", source, dest, "DDDDDDDD", idle_list, source_list, time.time(), u, ss, sp

							memcpy_p2p(source, dest, task)
							cur_dict[u][ss][sp].remove(task)
							if cur_dict[u][ss][sp] == []: del cur_dict[u][ss][sp]
							if cur_dict[u][ss] == {}: del cur_dict[u][ss]
							if cur_dict[u] == {}: del cur_dict[u]
							
							"""
							idle_list.remove(source)
							work_list[source] = []
							source_list.remove(source)
	
							if VIVALDI_BLOCKING and flag == 1:
								idle_list.remove(dest)
								if dest in source_list:
									source_list.remove(dest)
								work_list[dest] = []
							
							if source_list == []:
								flag_times["time1"] += time.time() - st
								return 
							"""
	return return_flag
def temp_func2(source_list=None): # dynamic function execution
	if source_list == [2]: return False
	cur_list = list(computing_unit_list)
#	if source_list == None or VIVALDI_BLOCKING: cur_list = list(idle_list)
#	else:
#		cur_list = [elem for elem in source_list if elem in idle_list]

	if 2 in cur_list: cur_list.remove(2)
	st = time.time()

	task_list = function_list
	return_flag = False
	for elem in list(task_list):
		if cur_list == []:break
		# check execid is available for this functions
		fp = elem
		reserved = fp.reserved

		execid_list = fp.execid_list
		t_list = [i for i in execid_list if i in cur_list]
		if t_list == []: continue

		args = fp.function_args
		flag = 0

		dp = None
		avail_set = {}
		for arg in args:
			u = arg.get_unique_id()
			if u == '-1': continue
			ss = arg.get_split_shape()
			sp = arg.get_split_position()

			# check data is assigned to be created
			rc = remaining_write_count_list
			if u not in rc:flag = 1
			elif ss not in rc[u]:flag = 1
			elif sp not in rc[u][ss]:flag = 1
		
			if flag == 1:
				dp = arg
				break
			# check there is usable data
			avail_set[u] = []

			for data_halo in remaining_write_count_list[u][ss][sp]:
				if data_halo < arg.data_halo: continue
				for source in remaining_write_count_list[u][ss][sp][data_halo]:
					if source not in avail_set[u] and remaining_write_count_list[u][ss][sp][data_halo][source] <= 0:
						avail_set[u].append(source)

			# check data is completed
			if avail_set[u] == []:
				flag = 2
				dp = arg
				break

			# find common set
			t = [i for i in t_list if i in avail_set[u]]
			t_list = t

		# data not exist
		# we have to make
		if flag == 1: continue

		# data creation is already started.
		# we can start this function if we wait
		if flag == 2: continue

		# all data exist 
		if VIVALDI_SCHEDULE == 'round_robin' and fp.reserved == False:
			if idle_list == [2]:return

			dest = get_round_robin(execid_list, idle=True)
			if dest == None: continue
			fp.dest = dest
			fp.reserved = True
			reserved = fp.reserved
			collect_data(args, dest)
			inform(function_package.output, dest=dest)
		
		# locality-aware
		# not in the same machine, or in the hard disk
		if t_list == [] or t_list == [2]:
			if reserved: continue
			# data is distributed all machines
			bytes_list = {}

			# check, available sources except harddisk
			for dp in args:
				u = dp.get_unique_id()
				if u == '-1': continue
				ss = dp.get_split_shape()
				sp = dp.get_split_position()
				n_data_halo = dp.data_halo

				rc = remaining_write_count_list[u][ss][sp]
				for source1 in computing_unit_list:
					for data_halo in rc:
						if data_halo < n_data_halo: continue
						if source1 not in rc[data_halo] or rc[data_halo][source1] > 0: continue
						if source1 not in bytes_list: bytes_list[source1] = 0
						bytes_list[source1] += dp.get_bytes()
						break

			max = 0
			dest = None
	
			# select maximum in round_robin	
			for elem in bytes_list:
				if elem == 2:continue
				val = bytes_list[elem]
				if max < val:
					max = val
					dest = elem

			# we have to execute at dest, but it is reserved by other function
			if dest != None and dest in reserved_dict:
				continue
			# data not exist

			cur_list = [elem for elem in execid_list if elem not in reserved_dict]
			if dest == None:
				# than just round lobin
				dest = get_round_robin(cur_list, idle=True)

			# destination is not idle
			if dest == None:
				continue

			collect_data(args, dest)
			reserved_dict[dest] = fp
			fp.reserved = True
			fp.dest = dest

			dp = fp.output
			u = dp.get_unique_id()
			ss = dp.get_split_shape()
			sp = dp.get_split_position()
			data_halo = dp.data_halo

			remaining_write_count_list[u][ss][sp][data_halo][dest] = 1
			flag_times["send_order"] += time.time()-st
	
			return_flag = True
		else:
			# there is a machine have all data need to run function
			if fp.reserved:
				if fp.dest not in t_list:continue
				t_list = [fp.dest]

				if fp.dest in reserved_dict:
					del reserved_dict[fp.dest]

			il = [elem for elem in t_list if elem in idle_list]
			if il != [] and il != [2]:
				dp = fp.output

				dest = get_round_robin(il)
				run_function(dest, fp)
				function_list.remove(fp)
				idle_list.remove(dest)
				work_list[dest] = []
				cur_list.remove(dest)
				
	return return_flag
def temp_func3():
	ft = 0
	return_flag = True

	for elem in data_list:
		data_package = elem[0]
		execid_list = elem[1]
		if idle_list == [] or idle_list ==[2]:break # reader cannot be destination of making data
		dp = data_package

		flag = make_buffer(dp, execid_list)
		if flag:
			data_list.remove(elem)
def temp_func4(source_list=None): # function execution
	if source_list == None or VIVALDI_BLOCKING: cur_list = list(idle_list)
	else:
		cur_list = [elem for elem in source_list if elem in idle_list]

	cur_list = list(idle_list)

	for function_package in list(function_list):
		if cur_list == []:break
		if function_package.dest not in cur_list: continue
		dest = function_package.dest
		fp = function_package
		args = fp.function_args
		flag = 0

		dp = None
		avail_set = {}
		for arg in args:
			u = arg.get_unique_id()
			if u == '-1': continue
			ss = arg.get_split_shape()
			sp = arg.get_split_position()

			# check data is assigned to be created
			if u not in remaining_write_count_list:flag = 1
			elif ss not in remaining_write_count_list[u]:flag = 1
			elif sp not in remaining_write_count_list[u][ss]:flag = 1
			if flag == 1:
				dp = arg
				break

			# check data is completed
			avail_set[u] = []
			rc = remaining_write_count_list[u][ss][sp]
			for halo in rc:
				if halo < arg.buffer_halo: continue
				source = dest
				if source not in avail_set[u] and source in rc[halo] and rc[halo][source] <= 0:
					avail_set[u].append(source)


			# check data is completed

			if avail_set[u] == []:
				flag = 2
				break

		# for end
		if flag == 1:
			# data not exist yet
			continue
		if flag == 2:
			continue

		dest = function_package.dest
		run_function(dest, function_package)
		function_list.remove(function_package)
		cur_list.remove(dest)
		idle_list.remove(dest)
		work_list[dest] = []

	return 
def launch_task(source_list=None):
	# launch every waiting functions
	if source_list == None or VIVALDI_BLOCKING:
		source_list = list(idle_list)
	if idle_list == []: return
	if VIVALDI_DYNAMIC:
		 temp_func3() # select destination of data
	if idle_list == []: return
	allow_memcpy = temp_func1(for_save=True, source_list=source_list) # memory copy for save to hardware
	if idle_list == []: return
	flag = False
	if 2 in source_list:
		flag = True
		source_list.remove(2)
	temp_func1(source_list=source_list) # memory copy for already assigned memory copy tasks
	if flag:
		source_list.append(2)
	if idle_list == []: return
	if VIVALDI_DYNAMIC: # dynamic allocation
		temp_func2(source_list=source_list) # launch available functions
	else: # static allocation
		temp_func4(source_list=source_list) # launch already reserved functions

	# from hard disk
	if 2 in idle_list:
		temp_func1(source_list=[2]) # memory copy for already assigned memory copy tasks
def get_round_robin(execid_list=None, idle=False):
	if execid_list == None:
		execid_list = computing_unit_list

	min = 0xffffff
	dest = None
	cur_list = [elem for elem in computing_unit_list if elem in execid_list]
	for source in cur_list:
		if idle and source not in idle_list:continue
		val = len(making_data[source])
		if min > val:
			dest = source
			min = val

	if dest == None:
		# no idle process
		return None

	computing_unit_list.remove(dest)
	computing_unit_list.append(dest)
	return dest

# task functions
def register_function(function_package):
	fp = function_package
	function_list.append(fp)
	
	args = fp.get_args()
	if fp.execid_list == []:
		fp.execid_list = idle_list+work_list.keys()

	for arg in args:
		
		flag = register_arg(arg, fp.execid_list)
		if VIVALDI_DYNAMIC: 
			if arg.get_unique_id() != '-1':
				inform(arg)
	
	if VIVALDI_DYNAMIC:
		inform(fp.output)

	if not VIVALDI_DYNAMIC:
		reserve_function(fp)
def run_function(dest, function_package):
	comm.isend(rank,                dest=dest,    tag=5)
	comm.isend("run_function",      dest=dest,    tag=5)
	comm.isend(function_package,    dest=dest,    tag=51)
	inform(function_package.output, dest=dest)

# memory management functions
def inform(data_package, dest=None, count=None):

	if not VIVALDI_DYNAMIC:
		if dest == None:
			return
	dp = data_package.copy()
	name			= dp.data_name
	u				= dp.get_unique_id()
	ss				= dp.get_split_shape()
	sp				= dp.get_split_position()

	if u not in data_packages: data_packages[u] = {}
	if ss not in data_packages[u]: data_packages[u][ss] = {}
	if sp not in data_packages[u][ss]: data_packages[u][ss][sp] = None

	data_halo = dp.data_halo
	rc = remaining_write_count_list
	if u not in rc: rc[u] = {}
	if ss not in rc[u]: rc[u][ss] = {}
	if sp not in rc[u][ss]: rc[u][ss][sp] = {}
	if data_halo not in rc[u][ss][sp]: rc[u][ss][sp][data_halo] = {}
	if dest not in rc[u][ss][sp][data_halo]:rc[u][ss][sp][data_halo][dest] = 1
	
	data_packages[u][ss][sp] = dp

	if u not in retain_count: retain_count[u] = {}
	if ss not in retain_count[u]: retain_count[u][ss] = {}
	if sp not in retain_count[u][ss]: retain_count[u][ss][sp] = {}
	if data_halo not in retain_count[u][ss][sp]: retain_count[u][ss][sp][data_halo] = 0
	
def register_arg(arg, execid_list = []):
	u, ss, sp = arg.get_id()
	if u == '-1': return False
	data_halo = arg.data_halo

	if execid_list == []: execid_list = idle_list+work_list.keys()

	
	flag = 0
	# check data is assigned to be created
	rc = remaining_write_count_list
	if u not in rc:flag = 1
	elif ss not in rc[u]:flag = 1
	elif sp not in rc[u][ss]:flag = 1
	#print "REGISTER ARGS", u, ss, sp, data_halo, flag
	if flag == 1: # data not exist
		retain(arg)
		# data_list for make memcpy tasks
		#data_list.append([arg, execid_list])
		# increase retain count of data sources
		source_list, SP_list, full_copy_range, count = increase_sources_retain_count(arg)
		source_list_dict[str((u,ss,sp,data_halo))] = source_list
		
		if VIVALDI_DYNAMIC:
			data_list.append([arg, execid_list, SP_list, full_copy_range, count])
		else:
			make_buffer(arg, execid_list, SP_list=SP_list, full_copy_range=full_copy_range, count=count)

		retain(arg)
		return True
		
	# check available buffer halo size exist for this data 
	for data_halo in rc[u][ss][sp]:
		if data_halo < arg.data_halo: continue
		for source in rc[u][ss][sp][data_halo]:
			retain(arg)
			return True
	
	retain(arg)
	# data is not assigned to be created
	# data_list for make memcpy tasks
	#data_list.append([arg, execid_list])

	data_halo = arg.data_halo

	# increase retain count of data sources
	source_list, SP_list, full_copy_range, count = increase_sources_retain_count(arg)
	source_list_dict[str((u,ss,sp,data_halo))] = source_list
	
	if VIVALDI_DYNAMIC:
		data_list.append([arg, execid_list, SP_list, full_copy_range, count])
	else:
		make_buffer(arg, execid_list, SP_list=SP_list, full_copy_range=full_copy_range, count=count)
	
	return True
def notice(data_package, source, source_package=None):
	dp = data_package
	name			= dp.data_name
	u, ss, sp = dp.get_id()
	data_halo		= dp.data_halo
	work_range = dp.data_range

	if source_package != None:
		mem_release(source_package)

	# check data_package come with real data 
	# it should not be with real data
	if dp.data != None or dp.devptr != None:
		assert(False)

	# check data is initialized
	def check_init(u,ss,sp):	
		rc = remaining_write_count_list
		if u in rc:
			if ss in rc[u]:
				if sp in rc[u][ss]:
					if data_halo in rc[u][ss][sp]:
						return True
		return False
	inited = check_init(u,ss,sp)

	#print dp.info(), data_halo, inited
	if inited == False:
		inform(dp, dest=source)
		
#	print_write_count()
#	print "======================="
#	print_retain_count()
	# real work from here
	remaining_write_count_list[u][ss][sp][data_halo][source] -= 1
	counter = remaining_write_count_list[u][ss][sp][data_halo][source]
	
#	print "source", source
#	print working_list
#	print data_package.info()
	if source in working_list:
		#print "AAAAAAAAAA", working_list
		for elem in list(working_list[source]):
			if elem.get_id() == dp.get_id():
				working_list[source].remove(elem)
		#print "BBBBBBBBBB", working_list, source
		if working_list[source] == []:
			del( working_list[source])
		#print "CCCCCCCCCC", counter
#	print "NOT", u, ss, sp, data_halo, source, counter
	if counter == 0 and source != main_unit_rank:
		if u not in valid_list: valid_list[u] = {}
		if ss not in valid_list[u]: valid_list[u][ss] = {}
		if sp not in valid_list[u][ss]: valid_list[u][ss][sp] = []
		valid_list[u][ss][sp].append(source)

		flag = False
		for elem in making_data[source]:
			mu = elem[0]
			mss = elem[1]
			msp = elem[2]
			mhalo = elem[3]

			if mu == u and mss == ss and msp == sp and mhalo == data_halo:
				del(elem)
				flag = True
				break

		if flag: making_data[source].remove((u,ss,sp,data_halo))

	launch_task(source_list=[source])
	if VIVALDI_DYNAMIC:
		launch_task()
def Free(u,ss,sp,data_halo):
	free_list = []

	not_valid = False
	rc = remaining_write_count_list[u][ss][sp]

	for execid in rc[data_halo]:
		if execid not in free_list:
			# check another buffer_halo retain_counter is remaining
			flag = True
			for other_halo in retain_count[u][ss][sp]:
				if other_halo == data_halo: continue
				if retain_count[u][ss][sp][other_halo] > 0:
					flag = False
					break

			if flag:
				not_valid = True
				free_list.append(execid)
	
	if not_valid:
		if u in valid_list:
			if ss in valid_list[u]:
				if sp in valid_list[u][ss]: 
					del valid_list[u][ss][sp]
					if valid_list[u][ss] == {}: del valid_list[u][ss]

	# let computing units free the volume
	for dest in free_list:
		if dest == 2: continue
		if dest != None:
			tag = 5
			comm.isend(rank,      dest=dest,	tag=tag)
			comm.isend("free",    dest=dest,	tag=tag)
			tag = int("%d%d"%(tag,8))
			comm.isend(u,         dest=dest,	tag=tag)
			comm.isend(ss,        dest=dest,	tag=tag)
			comm.isend(sp,        dest=dest,	tag=tag)
			comm.isend(data_halo, dest=dest,	tag=tag)

	#del(data_packages[u][ss][sp][halo])
		del(remaining_write_count_list[u][ss][sp][data_halo][dest])
	
	if remaining_write_count_list[u][ss][sp][data_halo] == {}: del(remaining_write_count_list[u][ss][sp][data_halo])
	if remaining_write_count_list[u][ss][sp] == {}: del(remaining_write_count_list[u][ss][sp])
	if remaining_write_count_list[u][ss] == {}: del(remaining_write_count_list[u][ss])
	if remaining_write_count_list[u] == {}: del(remaining_write_count_list[u])

	if free_list == [None] or free_list == []:
		return 

	if data_halo in retain_count[u][ss][sp]: 
		del(retain_count[u][ss][sp][data_halo])
	
	if retain_count[u][ss][sp] == {}: del(retain_count[u][ss][sp])
	if retain_count[u][ss] == {}: del(retain_count[u][ss])
	if retain_count[u] == {}: del(retain_count[u])

	target = (u,ss,sp,data_halo)
	# free related to function list
	#################################################
	for function_package in function_list:
		dp = function_package.output
		du, dss, dsp = dp.get_id()
		dh = dp.data_halo

		if target == (du,dss,dsp,dh):
			function_list.remove(function_package)
			arg_list = function_package.function_args
			for elem in arg_list:
				au = elem.get_unique_id()
				if au == '-1':continue
				release(elem)

	# free memcpy tasks
	##################################################
	for du in memcpy_tasks.keys():
		for dss in memcpy_tasks[du].keys():
			for dsp in memcpy_tasks[du][dss].keys():
				for task in memcpy_tasks[du][dss][dsp]:
					dp = task.dest
					dh = dp.data_halo
			
					if target == (du,dss,dsp,dh):
						memcpy_tasks[du][dss][dsp].remove(task)
						if memcpy_tasks[du][dss][dsp] == []: del memcpy_tasks[du][dss][dsp]
						if memcpy_tasks[du][dss] == {}: del memcpy_tasks[du][dss]
						if memcpy_tasks[du] == {}: del memcpy_tasks[du]
						release(task.source)

	for du in memcpy_tasks_to_harddisk.keys():
		for dss in memcpy_tasks_to_harddisk[du].keys():
			for dsp in memcpy_tasks_to_harddisk[du][dss].keys():
				for task in memcpy_tasks_to_harddisk[du][dss][dsp]:
					dp = task.dest
					dh = dp.data_halo

					if target == (du,dss,dsp,dh):
						memcpy_tasks_to_harddisk[du][dss][dsp].remove(task)
						if memcpy_tasks_to_harddisk[du][dss][dsp] == []: del memcpy_tasks_to_harddisk[du][dss][dsp]
						if memcpy_tasks_to_harddisk[du][dss] == {}: del memcpy_tasks_to_harddisk[du][dss]
						if memcpy_tasks_to_harddisk[du] == {}: del memcpy_tasks_to_harddisk[du]
						release(task.source)

	# free data_list
	##################################################
	for elem in data_list:
		dp = elem[0]
		du, dss, dsp = dp.get_id()
		dh = dp.data_halo

		if target == (du,dss,dsp,dh):
			data_list.remove(elem)
def release(data_package):
	dp = data_package

	u, ss, sp = dp.get_id()
	data_halo = dp.data_halo
	#print "Release", u, ss, sp, data_halo
	if ss == 'None': ss = str(SPLIT_BASE)
	if sp == 'None': sp = str(SPLIT_BASE)
	
	try:
		retain_count[u][ss][sp][data_halo] -= 1
	except:
		print "UU", u, ss, sp, data_halo
		print retain_count
		assert(False)
	
	if retain_count[u][ss][sp][data_halo] == 0:
		Free(u,ss,sp,data_halo)
def retain(data_package):
	dp = data_package

	u, ss, sp = dp.get_id()
	data_halo = dp.data_halo
	
	if u not in retain_count: retain_count[u] = {}
	if ss not in retain_count[u]: retain_count[u][ss] = {}
	if sp not in retain_count[u][ss]: retain_count[u][ss][sp] = {}
	if data_halo not in retain_count[u][ss][sp]: retain_count[u][ss][sp][data_halo] = 0

	retain_count[u][ss][sp][data_halo] += 1	
def make_a_memcpy_task(source_package, dest_package, dest, work_range, start=None):
	mt = Memcpy_task()
	# source
	mt.source = source_package
	# dest
	mt.execid = dest

	if dest == None:
		assert(False)

	# start
	mt.start = start

	# work_range
	mt.work_range = work_range

	# dest
	inform(dest_package, dest=dest)
	mt.dest = dest_package

	# append to memcpy task list
	u, ss, sp = mt.source.get_id()
	if dest == 2: 
		if u not in memcpy_tasks_to_harddisk: memcpy_tasks_to_harddisk[u] = {}
		if ss not in memcpy_tasks_to_harddisk[u]: memcpy_tasks_to_harddisk[u][ss] = {}
		if sp not in memcpy_tasks_to_harddisk[u][ss]: memcpy_tasks_to_harddisk[u][ss][sp] = []
		memcpy_tasks_to_harddisk[u][ss][sp].append(mt)
	else: 
		if u not in memcpy_tasks: memcpy_tasks[u] = {}
		if ss not in memcpy_tasks[u]: memcpy_tasks[u][ss] = {}
		if sp not in memcpy_tasks[u][ss]: memcpy_tasks[u][ss][sp] = []
		memcpy_tasks[u][ss][sp].append(mt)
def check_SP_exist(u, ss, SP_list):
	temp = []
	for sp in SP_list:
		sp = str(sp)
		return_flag = False
		if sp not in remaining_write_count_list[u][ss]:
			return False
		for halo in remaining_write_count_list[u][ss][sp]:
			source_list = remaining_write_count_list[u][ss][sp][halo].keys()
			if 2 in source_list:
				source_list.remove(2)
				source_list.append(2)
			for source in source_list:
				return_flag = True
				temp.append((u,ss,sp,halo,source))
				break
			if return_flag:break
		if return_flag == False:
			return False
	return temp
def increase_sources_retain_count(data_package):
	dp = data_package
	u, ss, sp = dp.get_id()
	data_halo = dp.data_halo
	rc = remaining_write_count_list			
	
	if u not in rc: rc[u] = {}
	
	for SS in rc[u]:
		SS = ast.literal_eval(SS) # data source
		boundary_list, full_copy_range, count = find_boundary_range(dp, SS)
		SP_list = make_SP_list(boundary_list)
		source_list = check_SP_exist(u, str(SS), SP_list)
		if source_list == False: continue
		for elem in source_list:
			sdp = Data_package()
			sdp.unique_id = elem[0]
			sdp.split_shape = elem[1]
			sdp.split_position = elem[2]
			sdp.data_halo = elem[3]

			# retain count of sources
			retain(sdp)
			
		
		if ss not in rc[u]: rc[u][ss] = {}
		if sp not in rc[u][ss]: rc[u][ss][sp] = {}
		if data_halo not in rc[u][ss][sp]: rc[u][ss][sp][data_halo] = {}
		rc[u][ss][sp][data_halo][None] = 1
			
	#	log("rank%d, u=%d, remaining_write_counter=%d"%(rank, u, count),'general',log_type)
		return source_list, SP_list, full_copy_range, count

	assert(False)

# support functions
def find_boundary_range(ss, sp, SS=None):
	count = 1
	boundary_list = {}
	boundary_value = {}

	dp = ss
	ss = ast.literal_eval(dp.get_split_shape())
	SS = sp

	if type(SS) == str:
		SS = ast.literal_eval(SS)
	fbrg = dp.full_data_range

	data_halo = dp.data_halo
	data_range = dp.data_range
	full_data_range = dp.full_data_range

	for axis in ss:
		if axis not in dp.data_range:
			boundary_list[axis] = (1,1)
			continue
			
		start = 0
		end = SS[axis]-1
		width = full_data_range[axis][1]-full_data_range[axis][0]
		left_boundary = data_range[axis][0]
		right_boundary = data_range[axis][1]
		boundary_value[axis] = (left_boundary, right_boundary)

		fb_left = full_data_range[axis][0]
		for n in range(SS[axis]):
			left = float(n)/SS[axis]*width + fb_left
			right = float(n+1)/SS[axis]*width + fb_left
		
			if left <= left_boundary: start = n
			if right >= right_boundary: 
				end = n
				break

		boundary_list[axis] = (start+1, end+1)
		count *= (end-start)+1
	
	return boundary_list, boundary_value, count
def make_SP_list(boundary_list):
	SP_list = []
	for axis in AXIS:
		temp = range(boundary_list[axis][0], boundary_list[axis][1]+1)
		temp2 = []
		if len(SP_list) == 0:
			for elem in temp:
				temp2.append({axis:elem})
		else:
			for front in SP_list:
				for back in temp:
					t = dict(front)
					t[axis] = back
					temp2.append(t)

		SP_list = temp2

	rt = []
	for elem in SP_list:
		rt.append(str(elem))

	return rt
def make_full_copy_range(u, SS, boundary_value):
	SP = data_packages[u][str(SS)].keys()[0]
	dp = data_packages[u][str(SS)][str(SP)]
	work_range = dict(dp.data_range)
	full_range = dict(dp.full_data_range)
	full_copy_range = dict(dp.full_data_range)

	for axis in work_range:
		s = full_copy_range[axis][0]
		w = full_copy_range[axis][1] - full_copy_range[axis][0]
		start = w*boundary_value[axis][0] + s
		end	= w*boundary_value[axis][1] + s

		full_copy_range[axis] = (int(start),int(end))
	return full_copy_range
def clip_range(valid_range, full_copy_range):
	for axis in valid_range:
		start = full_copy_range[axis][0]
		end	= full_copy_range[axis][1]
		start = start if valid_range[axis][0] < start else valid_range[axis][0]
		end = end if end < valid_range[axis][1] else valid_range[axis][1]
		valid_range[axis] = (int(start), int(end))

	return valid_range
def make_bytes_list(u, SS, SP_list, full_copy_range):
	copy_range_list = {}
	for SP in SP_list:
		sdp = data_packages[u][str(SS)][str(SP)]

		valid_range = get_valid_range(sdp.full_data_range, sdp.split_shape, sdp.split_position) # this range is always valid
		
		cliped_range = clip_range(valid_range, full_copy_range)
		
		copy_range_list[str(SP)] = make_bytes(cliped_range)

	return copy_range_list
def select_dest_by_byte_order(bytes_list, u, ss, SP_list, execid_list):
	bytes_with_rank = {}
	ss = str(ss)
	max = -1
	dest = -1
	rc = remaining_write_count_list
	for source in idle_list+work_list.keys():
		if source == 2: continue
		bytes_with_rank[source] = 0
		for sp in SP_list:
			sp = str(sp)
			for halo in rc[u][ss][sp]:
				if source not in rc[u][ss][sp][halo]:continue
#				if rc[u][ss][sp][halo][source] <= 0:
				bytes_with_rank[source] += bytes_list[sp]
				break
		
		bytes = bytes_with_rank[source]
		if max < bytes:
			max = bytes


	for source in execid_list:
#		if source not in idle_list:continue
		if source == 2:continue
		bytes = bytes_with_rank[source]
		if max == bytes:
			dest = source
			break

	if max == 0:
		dest = get_round_robin()
	return dest	

def check_source_is_available(source):
	u = source[0]
	ss = source[1]
	sp = source[2]
	halo = source[3]

	rc = remaining_write_count_list
	return_flag = False
	for source in rc[u][ss][sp][halo]:
		if source == None: continue

		if VIVALDI_DYNAMIC:
			if rc[u][ss][sp][halo][source] > 0:
				continue
		if source != 2:
			if u not in retain_count: continue
			if ss not in retain_count[u]: continue
			if sp not in retain_count[u][ss]: continue
			if halo not in retain_count[u][ss][sp]: continue
#			if retain_count[u][ss][sp][halo] <= 0:
#				continue
		return_flag = True
		break

	return return_flag 
def make_buffer(dp, execid_list, SP_list=None, full_copy_range=None, count=None):
	# make(u,ss,sp) data 
	u, ss, sp = dp.get_id()
	data_halo = dp.data_halo
	
	deleted = True
	if u in retain_count:
		if ss in retain_count[u]:
			if sp in retain_count[u][ss]:
				if data_halo in retain_count[u][ss][sp]:
					deleted = False

	if deleted:
		print "DELETED", u, ss, sp, data_halo
		print dp
		return True

	if u not in remaining_write_count_list: return False
	key = str((u,ss,sp,data_halo))
	if key not in source_list_dict: 
		print "key not in source_list_dict"
		print source_list_dict
		return False

	flag = False
	source_list = source_list_dict[key]

	for source in source_list:
		# check source is available now
		available = check_source_is_available(source)
		# if not return False
		if not available: 
			#print "source is not available"
			return False
	SS = source_list[0][1]
	
	if SP_list == None:
		SP_list = []
		for elem in source_list:
			SP_list.append(elem[2])

	if full_copy_range == None:
		boundary_list, full_copy_range, count = find_boundary_range(dp, SS)
	
	bytes_list = make_bytes_list(u, SS, SP_list, full_copy_range)
	
	dest = select_dest_by_byte_order(bytes_list, u, SS, SP_list, execid_list)

	if dest == -1:
		#print "DEST selection failed", dest
		#print bytes_list, u, SS, SP_list, execid_list
		return False
	
	
	for source in source_list:
		su = source[0]
		sss = source[1]
		ssp = source[2]
		shalo = source[3]
		sdp = data_packages[su][sss][ssp].copy()

		valid_range = get_valid_range(sdp.full_data_range, sdp.split_shape, sdp.split_position) # this range is always valid
		work_range = clip_range(valid_range, full_copy_range)
		dest_package = dp.copy()
	#	print "SOURCE", sdp
	#	print "DEST", dest_package, work_range
		make_a_memcpy_task(sdp, dest_package, dest, work_range)

	inform(dp, dest=dest)

	try:
		remaining_write_count_list[u][ss][sp][data_halo][dest] = count
	except:
		print "AAA", u, ss, sp, data_halo, dest
		print remaining_write_count_list
		assert(False)
	making_data[dest].append((u,ss,sp,data_halo))

	return True
def collect_data(arg_list=None, dest=None):

	for dp in arg_list:
		u, ss, sp = dp.get_id()
		if u == '-1': continue
		n_data_halo = dp.data_halo

		# check argument in dest
		flag = False
		rc = remaining_write_count_list[u][ss][sp]
		
		for data_halo in rc:
			if dest in rc[data_halo] and data_halo >= n_data_halo:
				flag = True
				break

		# already data exist 
		if flag:
			continue

		# not exist, we will make data copy
		source_package = dp.copy()
		source_package.buffer_range = None
		source_package.buffer_halo = None
		source_package.buffer_bytes = None
		retain(source_package)
		
		dest_package = dp.copy()
		work_range = dp.data_range
		make_a_memcpy_task(source_package, dest_package, dest, work_range)
		remaining_write_count_list[u][ss][sp][data_halo][dest] = 1
		inform(dp, dest=dest)
		
	return 
def print_write_count():
	rc = remaining_write_count_list
	for u in rc:
		for ss in rc[u]:
			for sp in rc[u][ss]:
				for data_halo in rc[u][ss][sp]:
					print u, ss, sp, data_halo, rc[u][ss][sp][data_halo]
def print_retain_count():
	for u in retain_count:
		for ss in retain_count[u]:
			for sp in retain_count[u][ss]:
				for data_halo in retain_count[u][ss][sp]:
					print u, ss, sp, data_halo, retain_count[u][ss][sp][data_halo]

# initialization
########################################################
def init_idle_list():
	global idle_list
	global comm
	size = comm.Get_size()
	# 0 main
	# 1 scheduler
	# 2 reader
	
	for i in range(2,size-1):
		idle_list.append(i)
	
init_idle_list()
for elem in idle_list: making_data[elem] = []
	
flag_times = {}
for elem in ["depth","release","retain","synchronize","merge","notice","inform","method","function","idle","send_order","synchronize","time1","time2","time3","time4","wait"]:
	flag_times[elem] = 0

# execution
########################################################
flag = ''
while flag != "finish":
#	print "Scheduler wait"
	if log_type != False: print "Scheduler wait"
	source = comm.recv(source=MPI.ANY_SOURCE,    tag=5)
	flag = comm.recv(source=source,              tag=5)
	if log_type != False: print "Scheduler source:", source, "flag:", flag
#	print "Scheduler source:", source, "flag:", flag
	# interactive mode functions
	if flag == "say":
		print "Scheduler hello", rank, name		
	elif flag == "process_status":
		comm.send(idle_list, dest=source, tag=5)
		comm.send(work_list, dest=source, tag=5)
	elif flag == "update_computing_unit":
		computing_unit_dict = comm.recv(source=source,    tag=5)
		computing_unit_list = computing_unit_dict.keys()
	elif flag == "log":
		log_type = comm.recv(source=source,    tag=5)
		print "Scheduler log type changed to", log_type
		for i in range(2,size):
			def change_log_type(log_type):
				dest = i
				tag = 5
				comm.isend(rank,        dest=dest,    tag=tag)
				comm.isend("log",       dest=dest,    tag=tag)
				comm.isend(log_type,    dest=dest,    tag=tag)
			change_log_type(log_type)
	elif flag == "remove_function":
		name = comm.recv(source=source,    tag=5)
		for i in range(3,size):
			def remove_function(name):
				dest = i
				tag = 5
				comm.isend(rank,                 dest=dest,    tag=tag)
				comm.isend("remove_function",    dest=dest,    tag=tag)
				comm.isend(name,                 dest=dest,    tag=tag)
			remove_function(name)
	elif flag == "get_data_list":
		for i in range(2,size):
			def get_data_list():
				dest = i
				tag = 5
				comm.isend(rank,               dest=dest,    tag=tag)
				comm.isend("get_data_list",    dest=dest,    tag=tag)
			get_data_list()
	elif flag == "remove_data":
		uid = comm.recv(source=source,    tag=5)
		for i in range(2,size):
			def remove_data():
				dest = i
				tag = 5
				comm.isend(rank,             dest=dest,    tag=tag)
				comm.isend("remove_data",    dest=dest,    tag=tag)
				comm.isend(uid,              dest=dest,    tag=tag)
			remove_data()
	# function initialization	
	if flag == "function":
		function_package = comm.recv(source=source, tag=52)
		
		def create_tasks(function_package):
			argument_package_list = function_package.get_function_args()
			def print_task_list(task_list):
				for task in task_list:
					print "TASK value TEST"
					print task.info()
					for elem in task.get_args():
						print elem.info()
					print task.output.info()
			def get_split_iter(full_range, split, halo):
				full_range = dict(full_range)
				split = eval(split)
				# calculate range list
				def get_range_list(full_range, split, halo):
					range_list = []
					for axis in AXIS:
						if axis in full_range:
							temp = []
							w = full_range[axis][1] - full_range[axis][0]
							n = split[axis] if axis in split else 1
							full_start = full_range[axis][0]
							for m in range(n): 
								start = m*w/n+full_start-halo
								if start < full_range[axis][0]: start = full_range[axis][0]
								end = (m+1)*w/n+full_start+halo
								if end > full_range[axis][1]: end = full_range[axis][1]
								
								temp.append( {axis:(start, end)} )
							
							if len(range_list) == 0:
								for elem in temp: range_list.append(elem)
							else:
								temp2 = []
								for elem2 in temp:
									for elem1 in range_list:
										t = {}
										t.update(elem1)
										t.update(elem2)
										temp2.append(t)
								range_list = temp2
					return range_list
				range_list = get_range_list(full_range, split, halo)
				def get_split_position(split_shape, n):
					if n == 0:
						print "I assumed n start from 1. if you want to set n as zero, please double check"
					p = 0
					split_position = dict(SPLIT_BASE)
					ss = split_shape
					sp = split_position
					sp[AXIS[0]] = n
					for axis in AXIS[0:-1]:
						q = sp[AXIS[p]]
						w = ss[AXIS[p]]
						a = int(q/w)

						if a*w == q: a -= 1
						sp[AXIS[p]] -= a*w
						sp[AXIS[p+1]] += a
						p += 1
					return split_position			
				# make split list
				split_iter = []
				i = 0
				n = len(range_list)
				for i in range(n):
					split_position = get_split_position(split, i+1)
					split_iter.append( (range_list[i], split_position))
				return split_iter	
			def in_and_out_check(argument_package_list):
				# check split is identical shape
				output_split_shape = function_package.output.get_split_shape()	
				def check_identical_shape(argument_package_list, split_shape):
					for argument_package in argument_package_list:
						if argument_package.get_unique_id() == '-1': continue
						if argument_package.get_split_shape() != split_shape: return False
					return True
				flag = check_identical_shape(argument_package_list, output_split_shape)
				if flag: return 'identical'

				# get in and out split count
				def get_split_cnt(split_shape):
					ss = eval(split_shape)
					cnt = 1
					for axis in ss:
						cnt *= ss[axis]
					return cnt
				out_split_cnt = get_split_cnt(function_package.output.get_split_shape())
				def get_input_split_cnt(argument_package_list):
					r_cnt = 1
					for argument_package in argument_package_list:
						if argument_package.get_unique_id() == '-1': continue
						r_cnt *= get_split_cnt(argument_package.get_split_shape())
					return r_cnt
				in_split_cnt = get_input_split_cnt(argument_package_list)
				
				if in_split_cnt != out_split_cnt: # check split number is same
					return False
					
				return 'different'
			def in_and_out1(task_list, argument_package_list):
				new_task_list = []
				for task in task_list:
					work_range = task.output.full_data_range
					halo = task.output.data_halo
					split_shape = task.output.get_split_shape()
					iter = get_split_iter(work_range, split_shape, halo)
					for split_elem in iter:
						def get_new_task(task, split_elem):
							work_range = split_elem[0]
							split_position = split_elem[1]
							
							new_task = copy.deepcopy(task)
							# task setting
							new_task.work_range = dict(work_range)
							
							# task output setting
							new_task.output.set_data_range(work_range)
							new_task.output.data_halo = halo
							new_task.output.split_position = split_position
							
							def input_argument_setting(argument_package_list, split_shape, split_position):
								for argument_package in argument_package_list:
									if argument_package.get_unique_id() == '-1': continue # constant
									if argument_package.get_split_shape() == str(SPLIT_BASE): continue # not split data
									argument_package.split_shape = split_shape
									argument_package.split_position = split_position
									argument_package.split_data_range(split_position)
								
							input_argument_setting(new_task.get_args(), split_shape, split_position)
							return new_task
							
						new_task = get_new_task(task, split_elem)
						new_task_list.append(new_task)
				return new_task_list
			def in_and_out2(task_list, argument_package_list):
				task = task_list[0]
				new_task_list = []
				n = len(argument_package_list)
				
				# prepare output split iter
				work_range = task.output.full_data_range
				halo = task.output.data_halo
				split_shape = task.output.get_split_shape()
				output_iter = get_split_iter(work_range, split_shape, halo)
				
				def recursive_task_argument_setting(argument_package_list, p):
					if p == n:
						def get_new_task(task, split_elem):
							work_range = split_elem[0]
							split_position = split_elem[1]
							
							new_task = task.copy()
							
							# copy argumnet
							new_task.set_args(copy.deepcopy(argument_package_list))
							
							# task setting
							new_task.work_range = dict(work_range)
							
							# task output setting
							new_task.output.set_data_range(work_range)
						#	new_task.output.data_halo = halo
							new_task.output.split_position = split_position
							
							return new_task
						cnt = len(new_task_list)
						split_elem = output_iter[cnt]
						new_task = get_new_task(task, split_elem)
						new_task_list.append(new_task)
						return 
					argument_package = argument_package_list[p]
					if argument_package.get_split_shape() != str(SPLIT_BASE):
						work_range = argument_package.full_data_range
						halo = argument_package.data_halo
						split = argument_package.get_split_shape()						
						iter = get_split_iter(work_range, split, halo)
						for split_elem in iter:
							work_range = split_elem[0]
							split_position = split_elem[1]
							
							argument_package.set_data_range(work_range)
							argument_package.data_halo = halo
							argument_package.split_position = split_position
							
							argument_package_list[p] = argument_package
							recursive_task_argument_setting(argument_package_list, p+1)
					else:
						recursive_task_argument_setting(argument_package_list, p+1)
				recursive_task_argument_setting(copy.deepcopy(argument_package_list), 0)
				
				return new_task_list
			def input_split(task_list, argument_package_list):
				fp = task_list[0]
				new_task_list = []
				n = len(argument_package_list)
				def recursive_task_argument_setting(argument_package_list, p):
					if p == n:
						# copy task
						new_task = copy.deepcopy(fp)
						
						# set task argument
						new_task.set_args(copy.deepcopy(argument_package_list))
						
						# append to task list
						new_task_list.append(new_task)
						
						return False
					argument_package = argument_package_list[p]
					if argument_package.get_split_shape() != str(SPLIT_BASE):
						work_range = argument_package.full_data_range
						halo = argument_package.data_halo
						split = argument_package.get_split_shape()						
						iter = get_split_iter(work_range, split, halo)
						for split_elem in iter:
							work_range = split_elem[0]
							split_position = split_elem[1]
							
							argument_package.set_data_range(work_range)
							argument_package.data_halo = halo
							argument_package.split_position = split_position
							
							fp.output.depth = make_depth(work_range, fp.mmtx)
							argument_package_list[p] = argument_package
							recursive_task_argument_setting(argument_package_list, p+1)
							# set input split flag
						return True
					else:
						return recursive_task_argument_setting(argument_package_list, p+1)
				sw = recursive_task_argument_setting(copy.deepcopy(argument_package_list), 0)
				# change id
				if sw: # mean input split occur
					ou_id = 0
					for new_task in new_task_list:
						new_id = new_task.output.get_unique_id() + '_input_split_' + str(ou_id)
						new_task.output.unique_id = new_id
						
						key = str(new_task.output.get_id())
						depth_dict[key] = new_task.output.depth
						
						ou_id += 1
						
				return new_task_list
			def output_split(task_list, argument_package_list):
				# output split change
				# 1. work range of task
				# 2. function output change with task range change
					
				new_task_list = []
				for task in task_list:
					work_range = task.output.full_data_range
					halo = task.output.data_halo
					split_shape = task.output.get_split_shape()
					iter = get_split_iter(work_range, split_shape, halo)
					for split_elem in iter:
						def get_new_task(task, split_elem):
							work_range = split_elem[0]
							split_position = split_elem[1]
							
							new_task = task.copy()
							# task setting
							new_task.work_range = dict(work_range)
							
							# task output setting
							new_task.output.set_data_range(work_range)
							new_task.output.data_halo = halo
							new_task.output.split_position = split_position
							
							return new_task
							
						new_task = get_new_task(task, split_elem)
						new_task_list.append(new_task)
				
				return new_task_list
			def prepare_modifier(function_package):
				def prepare_output(to):
					# split modifier
					split_shape = {} if to.split == None else to.split
					for axis in AXIS:
						if axis not in split_shape:
							split_shape[axis] = 1
					to.split_shape = dict(split_shape)
					to.split = None
					
					# halo modifier
					to.data_halo = to.halo
					to.halo = None
					return to
				def prepare_args(args_list):
					new_args = []
					for arg in args_list:
						if arg.get_unique_id() == '-1':
							new_args.append(arg)
							arg.split = None
							arg.halo = None
							continue
						# split modifier
						split_shape = {} if arg.split == None else arg.split
						for axis in AXIS:
							if axis not in split_shape:
								split_shape[axis] = 1
						arg.split_shape = dict(split_shape)
						arg.split = None
						
						# halo modifier
						arg.data_halo = arg.halo
						arg.halo = None
						
						new_args.append(arg)
					return new_args
					
				function_package.output = prepare_output(function_package.output)
				function_package.set_args(prepare_args(function_package.get_args()))
				
			prepare_modifier(function_package)
			
			task_list = [function_package]
			#print_task_list(task_list)
			
			flag = in_and_out_check(argument_package_list)
			#print "FLAG", flag
			if flag == 'identical':
				task_list = in_and_out1(task_list, argument_package_list)
			#	print_task_list(task_list)
				return task_list
			elif flag == 'different':
				task_list = in_and_out2(task_list, argument_package_list)
				return task_list
			else:			
				task_list = input_split(task_list, argument_package_list)
				task_list = output_split(task_list, argument_package_list)
				#print_task_list(task_list)
			
			return task_list
		task_list = create_tasks(function_package)
		# check task list
		if len(task_list) == 0:
			print "Vivaldi Warning"
			print "--------------------------------------"
			print "No task is created"
			print "--------------------------------------"
		# register task
		def register_tasks(task_list):
			for task in task_list:
				register_function(task)
		register_tasks(task_list)
		launch_task()
	elif flag == "set_function":
		Vivaldi_code = comm.recv(source=source, tag=5)
		for dest in computing_unit_list:
			tag = 5
			comm.isend(rank,           dest=dest, tag=tag)
			comm.isend('set_function', dest=dest, tag=tag)
			comm.isend(Vivaldi_code,   dest=dest, tag=tag)
	elif flag == "merge":
		st = time.time()
		
		data_package		= comm.recv(source=source, tag=53)
		rg					= comm.recv(source=source, tag=53)
		func_name			= comm.recv(source=source, tag=53)
		order				= comm.recv(source=source, tag=53)
		dimension			= comm.recv(source=source, tag=53)
		
		#Anukura
		front_back			= comm.recv(source=source, tag=53)

		dp = data_package
		u = dp.get_unique_id()
		log("rank%d, u=%d, \"%s\", %s, merge request come, make tasks for merge"%(rank, u, func_name, order),'general',log_type)
		function_compositing(data_package, rg, func_name, order, dimension, front_back)
		#function_compositing(data_package, rg, func_name, order, dimension)
		launch_task()
		flag_times[flag] += time.time()-st
	elif flag == "merge_new":
		function_package = comm.recv(source=source, tag=5)
		cnt              = comm.recv(source=source, tag=5)
		output_package = function_package.output
		# make merge task list
		# initialization
		new_task_list = []
		in_id = 0
		ou_id = cnt	
		# first merge
		def first_merge(cnt, in_id, ou_id):
			if cnt == 1: return cnt, in_id, ou_id
			for i in range(cnt/2):
				task = function_package.copy()
				argument_package_list = task.get_args()
				# set input argument id and calculate depth
				depth = 0
				for argument_package in argument_package_list:
					uid = argument_package.get_unique_id()
					if uid != '-1':
						new_id = argument_package.get_unique_id() + '_input_split_' + str(in_id)
						in_id += 1
						argument_package.unique_id = new_id
					
						# bring depth 
						key = str(argument_package.get_id())
						
						depth += depth_dict[key]
				def change_front_and_back(argument_package_list):
					cnt = 0
					ft = None
					bk = None
					# find first and back
					for argument_package in argument_package_list:
						uid = argument_package.get_unique_id()
						if uid != '-1':
							if cnt == 0:
								ft = argument_package
							else:
								bk = argument_package
								break
							cnt += 1
					# compare depth
					f_key = str(ft.get_id())
					b_key = str(bk.get_id())
					f_depth = depth_dict[f_key]
					b_depth = depth_dict[b_key]
					if f_depth > b_depth:
						temp = ft.unique_id
						ft.unique_id = bk.unique_id
						bk.unique_id = temp
						
						
				change_front_and_back(argument_package_list)
				depth /= 2
				# set depth
				task.output.depth = depth
				# set output argument id
				new_id = task.output.get_unique_id() + '_input_split_' + str(ou_id)
				ou_id += 1
				if cnt/2 > 1:
					task.output.unique_id = new_id
				# set depth_dict
				key = str(task.output.get_id())
				depth_dict[key] = depth
				# append to task
				new_task_list.append(task)
			cnt /= 2
			
			return cnt, in_id, ou_id
		cnt, in_id, ou_id = first_merge(cnt, in_id, ou_id)
		# second merge
		# change function_package
		def change_dtype():			
			task = function_package
			argument_package_list = task.get_args()
			op = task.output
			k = 0
			for argument_package in argument_package_list:
				uid = argument_package.get_unique_id()
				if uid != '-1':
					argument_package_list[k] = op.copy()
				k += 1
		change_dtype()
		def second_merge(cnt, in_id, ou_id):
			while cnt > 1:
				for i in range(cnt/2):
					task = function_package.copy()
					argument_package_list = task.get_args()
					# set input argument id and calculate depth
					depth = 0
					for argument_package in argument_package_list:
						uid = argument_package.get_unique_id()
						if uid != '-1':
							new_id = argument_package.get_unique_id() + '_input_split_' + str(in_id)
							in_id += 1
							argument_package.unique_id = new_id
							
							# bring depth 
							key = str(argument_package.get_id())
							depth += depth_dict[key]
					def change_front_and_back(argument_package_list):
						cnt = 0
						ft = None
						bk = None
						# find first and back
						for argument_package in argument_package_list:
							uid = argument_package.get_unique_id()
							if uid != '-1':
								if cnt == 0:
									ft = argument_package
								else:
									bk = argument_package
									break
								cnt += 1
						# compare depth
						f_key = str(ft.get_id())
						b_key = str(bk.get_id())
						f_depth = depth_dict[f_key]
						b_depth = depth_dict[b_key]
						
						if f_depth > b_depth:
							temp = ft.unique_id
							ft.unique_id = bk.unique_id
							bk.unique_id = temp
							
						print "FF", f_depth, b_depth, ft.unique_id, bk.unique_id
						

					change_front_and_back(argument_package_list)
							
					depth /= 2
					# set depth
					task.output.depth = depth
					# set output argument id
					new_id = task.output.get_unique_id() + '_input_split_' + str(ou_id)
					ou_id += 1
					if cnt/2 > 1:
						task.output.unique_id = new_id
					# set depth_dict
					key = str(task.output.get_id())
					depth_dict[key] = depth
					# append to task
					new_task_list.append(task)
				cnt /= 2
		second_merge(cnt, in_id, ou_id)
		# register task list
		def register_tasks(task_list):
			for task in task_list:
				register_function(task)	
		register_tasks(new_task_list)
		launch_task()
		# memory management
	if flag == "release":
		data_package = comm.recv(source=source, tag=57)
		dp = data_package
		release(dp)
	elif flag == "retain":
		data_package = comm.recv(source=source,  tag=58)		
		retain(data_package)
	elif flag == "synchronize":
		st = time.time()
		synchronize_flag = True
		synchronize()
		flag_times[flag] += time.time()-st
	elif flag == "finish":
		disconnect()
	elif flag == "notice":
		st = time.time()
		data_package    = comm.recv(source=source, tag=51)
		notice(data_package, source)
		flag_times[flag] += time.time()-st
	elif flag == "inform":
#		st = time.time()
#		data_package = comm.recv(source=source, tag=55)
#		dest = comm.recv(source=source, tag=55)
#		print "INFOM", data_package.info()
#		inform(data_package, source, dest=dest)
#		flag_times[flag] += time.time()-st
		pass
	elif flag == "method":
		st = time.time()
		u	= comm.recv(source=source,	tag=56)
		m	= comm.recv(source=source,	tag=56)
		compositing_method[u] = m
		log("rank%d, u=%d method %s"%(rank, u, m),'general',log_type) 
		flag_times[flag] += time.time()-st
	elif flag == "idle":
		log("WORK %s and IDlE%s"%(str(work_list.keys()), str(idle_list)),'progress', log_type)
		st = time.time()
		if source not in idle_list:
			idle_list.append(source)
		if source in work_list:
			del(work_list[source])

		log("IDlE %d"%(source),'progress', log_type)	
		log("WORK %s and IDlE%s"%(str(work_list.keys()), str(idle_list)),'progress', log_type)
		log("rank%d, rank%d is now idle"%(rank, source),'general', log_type)
		launch_task(source_list=[source])
		st2 = time.time()
		synchronize()
		flag_times["synchronize"] += time.time()-st2
		flag_times[flag] += time.time()-st
	elif flag == "notice_data_out_of_core":
		data_package = comm.recv(source=source, tag=501)
		notice(data_package, 2)

		dest = 2
		comm.isend(rank,						 dest=dest,	   tag=5)
		comm.isend("notice_data_out_of_core",	 dest=dest,	   tag=5)
		comm.isend(data_package,				 dest=dest,	   tag=501)
		
	elif flag == "save_image_out_of_core":
		data_package = comm.recv(source=source, tag=510)
		dp = data_package
		u = dp.get_unique_id()

		halo = dp.buffer_halo
		dp.set_halo(0)

		full_data_range = dp.full_data_range
		split_shape = ast.literal_eval(dp.split_shape)
		data_range_list = make_range_list(full_data_range, split_shape)
		
		n = 1

		for data_range in data_range_list:
			split_position = make_split_position(split_shape, n)
			dp.split_position = str(split_position)

			source_package = dp.copy()
			source_package.set_data_range(data_range)
			source_package.set_halo(halo)
			dest_package = dp.copy()
			dest_package.set_data_range(data_range)
			dest_package.set_halo(halo)

			retain(source_package)
			make_a_memcpy_task(source_package, dest_package, 2, apply_halo(data_range, halo))
			n += 1

		launch_task()

		comm.isend(rank, dest=2, tag=5)
		comm.isend("prepare_out_of_core_save", dest=2, tag=5)
		send_data_package(dp, dest=2, tag=502)
	elif flag == "save_image_in_core":
		data_package	= comm.recv(source=source, tag=511)
		dp = data_package
		u = dp.get_unique_id()
		
		dp.split_shape = str(SPLIT_BASE)
		dp.split_position = str(SPLIT_BASE)
		data_range = dp.full_data_range

		retain(dp)
		flag = register_arg(dp, idle_list+work_list.keys())
		dest_package = dp.copy()

		halo = dest_package.buffer_halo
		data_halo = dest_package.data_halo

		# data_range 
		data_range = dest_package.data_range
		dest_package.set_data_range(data_range)

		# full_data_range
		full_data_range = dest_package.full_data_range
		dest_package.set_full_data_range(full_data_range)

		make_a_memcpy_task(dp, dest_package, 2, data_range)
		launch_task()
	elif flag == "gather":
		data_package = comm.recv(source=source, tag=512)
		
		dp = data_package
		u = dp.get_unique_id()
		buffer_halo = dp.buffer_halo

		dp.split_shape = str(SPLIT_BASE)
		dp.split_position = str(SPLIT_BASE)
		data_range = dp.full_data_range
		dp.set_data_range(dp.full_data_range)

		flag = register_arg(dp.copy(), idle_list+work_list.keys()) # source retain
		
		source_package = dp.copy()
		dest_package = dp.copy()
		
		buffer_halo = dest_package.buffer_halo
		data_halo = dest_package.data_halo

		# data_range
		data_range = dest_package.data_range
		make_a_memcpy_task(source_package, dest_package, source, data_range)
		launch_task()
	elif flag == "reduce":
		data_package = comm.recv(source=source, tag=5)
		function_name = comm.recv(source=source, tag=5)
		return_package = comm.recv(source=source, tag=5)
		
		
		