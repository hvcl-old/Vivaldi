from mpi4py import MPI

# mpi init
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

def disconnect():
	print "AA", rank, name
	comm.Barrier()
	comm.Disconnect()
	exit()

from Vivaldi_load import *
from Vivaldi_misc import *
from Vivaldi_memory_packages import *


data_list = [] #which computing unit have data
source_list_dict = {}
data_packages = {}

remaining_write_count_list = {}
valid_list = {}

retain_count = {}
dirty_dict = {}

computing_unit_dict = {}
processor_list = []
idle_list = []
work_list = {}
reserved_dict = {}

# work list will divide to kernel running and data copy
kerner_list = {}
copy_list = {}

making_data = {}

depth_list = {}
compositing_method = {}

function_list = []
memcpy_tasks = {}
memcpy_tasks_to_harddisk = {}

synchronize_flag = False

cnt = 0

global muid
muid = -2

# misc function
###############################################################################
def print_retain_count():
	for u in retain_count:
		for ss in retain_count[u]:
			for sp in retain_count[u][ss]:
				for buffer_halo in retain_count[u][ss][sp]:
					print u, ss, sp, buffer_halo, retain_count[u][ss][sp][buffer_halo]

def print_write_count():
	rc = remaining_write_count_list
	for u in rc:
		for ss in rc[u]:
			for sp in rc[u][ss]:
				for data_halo in rc[u][ss][sp]:
					print u, ss, sp, buffer_halo, rc[u][ss][sp][buffer_halo]
					
def print_data_package():
	for u in data_packages:
		for ss in data_packages[u]:
			for sp in data_packages[u][ss]:
				print u, ss, sp, data_packages[u][ss][sp]
					
def send_data_package(data_package, dest=None, tag=None):
	dp = data_package
	t_data, t_devptr = dp.data, dp.devptr
	dp.data, dp.devptr = None, None
	comm.isend(dp, dest=dest, tag=tag)
	dp.data, dp.devptr = t_data, t_devptr
	t_data,t_devptr = None, None
	
# computing unit function
###############################################################################
def isCPU(execid):
	if execid in computing_unit_dict:
		if computing_unit_dict[execid]['device_type'] == 'CPU':return True
		return False

def isGPU(execid):
	if execid in computing_unit_dict:
		if computing_unit_dict[execid]['device_type'] == 'GPU':return True
		return False

def run_function(dest, function_package):
	comm.isend(rank,				dest=dest,   tag=5)
	comm.isend("run_function",	dest=dest,   tag=5)
	comm.isend(function_package,	dest=dest,	tag=51)
	log("Run funcion %d"%(dest), "progress", log_type)
#	print "RF", dest, time.time()
	inform(function_package.output, dest=dest)

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

#	print "Computing unit list", dest, computing_unit_list
	computing_unit_list.remove(dest)
	computing_unit_list.append(dest)
	return dest

# memory managemant funtions
##########################################################################function_list == [] and 
def inform(data_package, source=2, dest=None, count=None):

	if not VIVALDI_DYNAMIC:
		if dest == None:
			return
	dp = data_package.copy()
	name			= dp.data_name
	u				= dp.unique_id
	ss				= dp.split_shape
	sp				= dp.split_position

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
	log("rank%d, \"%s\", u=%d informed from rank%d"%(rank, name, u, source),'general', log_type)

#def temp_func1(for_save=False):
def temp_func1(for_save=False, source_list=None):
	st = time.time()
	return_flag = True

	if for_save: cur_dict = memcpy_tasks_to_harddisk
	else: cur_dict = memcpy_tasks

	source_list = [elem for elem in source_list if elem in idle_list]
	if source_list == []:return False

	if 3 in source_list:
		source_list.remove(3)
		source_list.append(3)

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
	flag_times["time1"] += time.time() - st
	return return_flag

#def temp_func2():
def temp_func2(source_list=None):

	if source_list == [3]: return False
	st = time.time()
	cur_list = list(computing_unit_list)
#	if source_list == None or VIVALDI_BLOCKING: cur_list = list(idle_list)
#	else:
#		cur_list = [elem for elem in source_list if elem in idle_list]

	if 3 in cur_list: cur_list.remove(3)	
	st = time.time()

	task_list = function_list

	return_flag = False

	for elem in list(task_list):
		if cur_list == []:break
		# check execid is avaiable for this functions
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
			u = arg.unique_id
			if u == -1: continue
			ss = arg.split_shape
			sp = arg.split_position

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

#			print "TTTTT", u, ss, sp
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
			if idle_list == [3]:return

			dest = get_round_robin(execid_list, idle=True)
			if dest == None: continue
			fp.dest = dest
			fp.reserved = True
			reserved = fp.reserved
			collect_data(args, dest)
			inform(function_package.output, dest=dest)

		
		# locality
		# not in the same machine, or in the hard disk
		if t_list == [] or t_list == [3]:
			if reserved: continue
			# data is distributed all machines
			bytes_list = {}

			# check, available sources except harddisk
			for dp in args:
				u = dp.unique_id
				if u == -1: continue
				ss = dp.split_shape
				sp = dp.split_position
				n_data_halo = dp.data_halo

				rc = remaining_write_count_list[u][ss][sp]
				for source1 in computing_unit_list:
					for data_halo in rc:
						if data_halo < n_data_halo: continue
						if source1 not in rc[data_halo] or rc[data_halo][source1] > 0: continue
						if source1 not in bytes_list: bytes_list[source1] = 0
						bytes_list[source1] += dp.data_bytes
						break

			max = 0
			dest = None
	
			# select maximum in round_robin	
			for elem in bytes_list:
				if elem == 3:continue
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
			u = dp.unique_id
			ss = dp.split_shape
			sp = dp.split_position
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
			if il != [] and il != [3]:
				dp = fp.output
				u = dp.unique_id
				ss = dp.split_shape
				sp = dp.split_position
				halo = dp.buffer_halo

				dest = get_round_robin(il)
				run_function(dest, fp)
				function_list.remove(fp)
				idle_list.remove(dest)
				work_list[dest] = []
				cur_list.remove(dest)
				break
				
	flag_times["time2"] += time.time() - st

	return return_flag

def temp_func3():
	st = time.time()
	ft = 0
	return_flag = True

	for elem in data_list:
		data_package = elem[0]
		execid_list = elem[1]
		if idle_list == [] or idle_list ==[3]:break # reader cannot be destination of making data
		dp = data_package

		flag = make_buffer(dp, execid_list)
		if flag:
			data_list.remove(elem)

	flag_times["time3"] += time.time() - st

def temp_func4(source_list=None):

	st = time.time()
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
			u = arg.unique_id
			if u == -1: continue
			ss = arg.split_shape
			sp = arg.split_position

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

	flag_times["time4"] += time.time() - st
	return 

def launch_task_before(source_list=None):
	"""
	launch every waiting functions
	"""
#	global synchronize_flag
#	if synchronize_flag == False: return

	if log_type in ['time','all']:
		st = time.time()

	repeat_flag = True

	while repeat_flag:
		repeat_flag = False
		if idle_list == []:continue

		allow_memcpy = temp_func1(for_save=True, source_list=source_list) # memory copy for save to hardware
		if idle_list == []:continue

		temp_func1(source_list=source_list) # memory copy for already assigned memory copy tasks
		if idle_list == []:continue

		if VIVALDI_DYNAMIC:

			repeat_flag = temp_func2(source_list=source_list) # launch available functions
			if repeat_flag: 
				continue

			# from hard disk
			if 3 in idle_list:
				temp_func1(source_list=[3]) # memory copy for already assigned memory copy tasks

			if idle_list == []:continue
			repeat_flag = temp_func3() # select destination of data

		else:
			temp_func4(source_list=source_list) # launch already reserved functions

			# from hard disk
			if 3 in idle_list:
				temp_func1(source_list=[3]) # memory copy for already assigned memory copy tasks


	if log_type in ['time','all']:
		elapsed = time.time() - st

def launch_task(source_list=None):
	"""
	launch every waiting functions
	"""
	if log_type in ['time','all']:
		st = time.time()
	
	if source_list == None or VIVALDI_BLOCKING:
		source_list = list(idle_list)

	if idle_list == []: return

	if VIVALDI_DYNAMIC:
		 temp_func3() # select destination of data

	if idle_list == []: return
	allow_memcpy = temp_func1(for_save=True, source_list=source_list) # memory copy for save to hardware
	
	if idle_list == []: return
	flag = False
	if 3 in source_list:
		flag = True
		source_list.remove(3)
	temp_func1(source_list=source_list) # memory copy for already assigned memory copy tasks
	if flag:
		source_list.append(3)

	if idle_list == []: return
	if VIVALDI_DYNAMIC: # dynamic allocation
		temp_func2(source_list=source_list) # launch available functions
	else: # static allocation
		temp_func4(source_list=source_list) # launch already reserved functions

	# from hard disk
	if 3 in idle_list:
		temp_func1(source_list=[3]) # memory copy for already assigned memory copy tasks

	if log_type in ['time','all']:
		elapsed = time.time() - st

def memcpy_p2p(source, dest, task):
	comm.isend(rank,             	dest=source,		tag=5)
	comm.isend("memcpy_p2p_send",	dest=source,		tag=5)
	comm.isend(dest,				dest=source,		tag=56)
	comm.isend(task,				dest=source,		tag=56)
	
def check_SP_exist(u, ss, SP_list):
	temp = []
	for sp in SP_list:
		sp = str(sp)
		return_flag = False
		if sp not in remaining_write_count_list[u][ss]:
			return False
		for halo in remaining_write_count_list[u][ss][sp]:
			source_list = remaining_write_count_list[u][ss][sp][halo].keys()
			if 3 in source_list:
				source_list.remove(3)
				source_list.append(3)
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
	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
	data_halo = dp.data_halo

	rc = remaining_write_count_list

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
			
		if u not in rc: rc[u] = {}
		if ss not in rc[u]: rc[u][ss] = {}
		if sp not in rc[u][ss]: rc[u][ss][sp] = {}
		if data_halo not in rc[u][ss][sp]: rc[u][ss][sp][data_halo] = {}
		rc[u][ss][sp][data_halo][None] = 1
		
		#print "EEE", u, ss, sp, data_halo, full_copy_range
		log("rank%d, u=%d, remaining_write_counter=%d"%(rank, u, count),'general',log_type)
		return source_list, SP_list, full_copy_range, count

	assert(False)

def register_arg(arg, execid_list = []):
	u = arg.unique_id
	if u == -1: return False
	ss = arg.split_shape
	sp = arg.split_position
	data_halo = arg.data_halo

	if execid_list == []: execid_list = idle_list+work_list.keys()

	#print "REGISTER ARGS", u, ss, sp, data_halo

	flag = 0
	# check data is assigned to be created
	rc = remaining_write_count_list
	if u not in rc:flag = 1
	elif ss not in rc[u]:flag = 1
	elif sp not in rc[u][ss]:flag = 1

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

def collect_data(arg_list=None, dest=None):

	for dp in arg_list:
		u = dp.unique_id
		if u == -1: continue
		ss = dp.split_shape
		sp = dp.split_position
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

def reserve_function(function_package):

	execid = None

	fp = function_package
	execid_list = fp.execid_list
	if execid_list == []: execid_list = idle_list+work_list.keys()
	t_list = list(execid_list)

	args = fp.function_args
	flag = 0

	if VIVALDI_SCHEDULE == 'round_robin':
		dest = get_round_robin(execid_list)
		fp.dest = dest
		collect_data(args, dest=dest)
		inform(function_package.output, dest=dest)	
		return 

	dp = None
	avail_set = {}
	for arg in args:
		u = arg.unique_id
		if u == -1: continue
		ss = arg.split_shape
		sp = arg.split_position

		# check data is completed
		avail_set[u] = []
		for halo in remaining_write_count_list[u][ss][sp]:
			if halo < arg.buffer_halo: continue
			for source in remaining_write_count_list[u][ss][sp][halo]:
				if source not in avail_set[u] and source != None:
					avail_set[u].append(source)

		# find common set
		t = [i for i in t_list if i in avail_set[u]]
		t_list = t


	dp = fp.output
	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
	halo = dp.buffer_halo

	if t_list == [] or t_list == [3]:
		# data is distirbuted all machines
#		execid_list = oe # oe is idle and avilable list for this function
		bytes_list = {}

		# check, all sources are available
		# make bytes_list of each computing unit

		for dp in args:
			u = dp.unique_id
			if u == -1: continue
			ss = dp.split_shape
			sp = dp.split_position
			buffer_halo = dp.buffer_halo

			sl = remaining_write_count_list[u][ss][sp]
			counted = {}
			for halo1 in sl:
				if halo1 < buffer_halo:continue
				for elem in sl[halo1]:
					if elem in counted: continue
					counted[elem] = 1
					if elem not in bytes_list:
						bytes_list[elem ] =0
					bytes_list[elem] += dp.data_bytes

		# select maximum dest
		max = 0
		dest = None

		if False:
			# high utilization
			cur_list = []
			for elem in bytes_list:
				if elem == 3:continue
				val = bytes_list[elem]
				if max < val:
					max = val
					dest = elem
					cur_list = [dest]
				elif max == val:
					cur_list.append(elem)
	
			# data not exist
			if dest == None:
				# than just round lobin
				dest = get_round_robin(execid_list)
			elif len(cur_list) > 1:
				dest = get_round_robin(cur_list)
		if True:
			# low number of utilization
			for elem in bytes_list:
				if elem == 3:continue
				val = bytes_list[elem]
				if max < val:
					max = val
					dest = elem
	
			# data not exist
			if dest == None:
				# than just round lobin
				dest = get_round_robin(execid_list)
	
		# collect input arguments at dest
		collect_data(arg_list=args, dest=dest)

		# let shceduler know, output will be created at dest
		inform(function_package.output, dest=dest)
		function_package.dest = dest
	else:
		# there is a machine have all data need to run function

		func_flag = False
		flag = False
		if 3 in t_list: 
			t_list.remove(3)
			flag = True
		dest = get_round_robin(t_list)
		inform(function_package.output, dest=dest)
		function_package.dest = dest # this functions reserved for this execid
	
		if flag:
			t_list.append(3)
		
	#	for execid in t_list:
	#		if execid == 3:continue
	#		#run_function(execid, function_package)
			# just reserve

#			inform(function_package.output, dest=execid)
#			function_package.dest = execid # this functions reserved for this execid
#			break

	return execid

def register_function(function_package):
	fp = function_package
	function_list.append(fp)
	log("rank%d, \"%s\", function registered"%(rank, fp.function_name),'general', log_type)

	args = fp.function_args
	if fp.execid_list == []:
		fp.execid_list = idle_list+work_list.keys()

	for arg in args:
		flag = register_arg(arg, fp.execid_list)
		if VIVALDI_DYNAMIC: 
			if arg.unique_id != -1:
				inform(arg)
	
	if VIVALDI_DYNAMIC:
		inform(fp.output)

	if not VIVALDI_DYNAMIC:
		reserve_function(fp)

def find_boundary_range(ss, sp, SS=None):
	count = 1
	boundary_list = {}
	boundary_value = {}

	dp = ss
	ss = ast.literal_eval(dp.split_shape)
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
		if source == 3: continue
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
		if source == 3:continue
		bytes = bytes_with_rank[source]
		if max == bytes:
			dest = source
			break

	if max == 0:
		dest = get_round_robin()
	return dest

# start is data copy start point
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
	u = mt.source.unique_id
	ss = mt.source.split_shape
	sp = mt.source.split_position

	if dest == 3: 
		if u not in memcpy_tasks_to_harddisk: memcpy_tasks_to_harddisk[u] = {}
		if ss not in memcpy_tasks_to_harddisk[u]: memcpy_tasks_to_harddisk[u][ss] = {}
		if sp not in memcpy_tasks_to_harddisk[u][ss]: memcpy_tasks_to_harddisk[u][ss][sp] = []
		memcpy_tasks_to_harddisk[u][ss][sp].append(mt)
	else: 
		if u not in memcpy_tasks: memcpy_tasks[u] = {}
		if ss not in memcpy_tasks[u]: memcpy_tasks[u][ss] = {}
		if sp not in memcpy_tasks[u][ss]: memcpy_tasks[u][ss][sp] = []
		memcpy_tasks[u][ss][sp].append(mt)

def make_tasks(u, SS, SP_list, dp, dest, full_copy_range ):
	for SP in SP_list:
		sdp = data_packages[u][str(SS)][str(SP)]
		work_range = data_range_minus_halo(sdp.data_range, sdp.buffer_halo)
		
		work_range = clip_range(work_list, full_copy_range)
		
		dest_package = dp.copy()
		dest_package.set_data_range(dict(full_copy_range))
		dest_package.set_full_data_range(dict(full_copy_range))
		make_a_memcpy_task(sdp, dest_package, dest, work_range)

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
		if source != 3:
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
	"""
		make(u,ss,sp) data 
	"""

	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
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

def make_a_task(name, dimension, dp, ss, sp, work_range, full_copy_range):

	# make proper function package
	function_package = Function_package()
	fp = function_package
	fp.set_function_name(name)

	function_args = [dp]
	if dimension >= 1:
		dpx = Data_package()
		dpx.data_name = 'x'
		dpx.unique_id = -1
		dpx.data_dtype = int
		function_args += [dpx]

	if dimension >= 2:
		dpy = Data_package()
		dpy.data_name = 'y'
		dpy.unique_id = -1
		dpy.data_dtype = int
		function_args += [dpy]

	if dimension >= 3:
		dpz = Data_package()
		dpz.data_name = 'z'
		dpz.unique_id = -1
		dpz.data_dtype = int
		function_args += [dpz]

	execid_list = [4]

	output = dp.copy()
	output.split_shape = ss
	output.split_position = sp

	output.data_range = full_copy_range #full_range

	output.data_shape = range_to_shape(full_copy_range) #output.full_data_shape
	cs = output.data_contents_memory_shape
	if cs == [1]: output.data_memory_shape = output.data_shape
	else: output.data_memory_shape = tuple(list(output.data_shape) + cs)

	bytes = 1
	for elem in output.data_memory_shape:
		bytes *= elem
	output.data_bytes = bytes*4

	fp.work_range = work_range
	fp.function_args = function_args
	fp.execid_list = execid_list
	fp.output = output

	inform(fp.output)
	
	register_function(fp)

#Anukura
#def function_compositing(data_package, rg, func_name, func_order, dimension):
def function_compositing(data_package, rg, func_name, func_order, dimension, front_back):
	dp = data_package
	def merge(out, in1, in2):
		fp = Function_package()
		fp.set_function_name(func_name)
		fp.output = dp.copy()

		fp.output.unique_id = out

		fp.work_range = fp.output.data_range
		
		#Anukura
		fp.front_back = front_back

		arg_package = []

		dp1 = data_packages[in1][str(SPLIT_BASE)][str(SPLIT_BASE)]
		dp2 = data_packages[in2][str(SPLIT_BASE)][str(SPLIT_BASE)]

		fp.output.depth = (dp1.depth + dp2.depth)/2

		dpx = Data_package()
		dpx.data_name = 'x'
		dpx.unique_id = -1
		dpx.data_dtype = int

		dpy = Data_package()
		dpy.data_name = 'y'
		dpy.unique_id = -1
		dpy.data_dtype = int

		arg_package = [dp1,dp2, dpx,dpy]
	
		if dimension == 3:
			dpz = Data_package()
			dpz.data_name = 'z'
			dpz.unique_id = -1
			dpz.data_dtype = int
			arg_package += [dpz]

		fp.function_args = arg_package


		inform(fp.output)
		register_function(fp)

	# select proper algorithm
	# we assume binary tree is proepr

	width = 1
	n = len(rg)
	s = str(SPLIT_BASE)

	flag = True
	if func_order == 'front-to-back': flag = True
	else: flag = False

	for w in range(1,n)[::-1]:
		d1 = data_packages[u+w*2-1][s][s].depth
		d2 = data_packages[u+w*2][s][s].depth

		if (d1 <= d2 and flag == True) or (d1 > d2 and flag == False):
			merge(u+w-1, u+w*2-1, u+w*2)
		else:
			merge(u+w-1, u+w*2, u+w*2-1)

def synchronize():
	global synchronize_flag

	if synchronize_flag == True and function_list == [] and memcpy_tasks == {} and work_list == {} and data_list == [] and memcpy_tasks_to_harddisk == {}:
		for dest in idle_list:
			comm.isend(rank,			dest=dest,	tag=5)
			comm.isend("synchronize",	dest=dest,	tag=5)
			comm.recv(source=dest, tag=999)
		
		comm.isend(True, dest=1, tag=999)
		synchronize_flag = False

def notice(data_package, source, source_package=None):
	dp = data_package
	name			= dp.data_name
	u				= dp.unique_id
	ss				= dp.split_shape
	sp				= dp.split_position
	data_halo		= dp.data_halo
	work_range = dp.data_range


	if source_package != None:
		mem_release(source_package)

	# check data_package come with real data 
	# it should not be with real data
	if dp.data != None or dp.devptr != None:
		assert(False)

	# check is this data already freed
	deleted = True

	if u in retain_count:
		if ss in retain_count[u]:
			if sp in retain_count[u][ss]:
				if data_halo in retain_count[u][ss][sp]:
					deleted = False

	if deleted:
		print "DELETED", dp, source
		assert(False)
#		print_retain_count()
		dest = source
		if dest != 3:
			tag = 0
			if isCPU(dest): tag = 4
			if isGPU(dest): tag = 5
			comm.isend(rank,		dest=dest,	tag=tag)
			comm.isend("free",	dest=dest,	tag=tag)
			tag = int("%d%d"%(tag,8))
			comm.isend(u,		dest=dest,	tag=tag)
			comm.isend(ss,		dest=dest,	tag=tag)
			comm.isend(sp,		dest=dest,	tag=tag)
			comm.isend(data_halo,dest=dest,	tag=tag)
		return True
	
	# real work from here
#	inform(dp, dest=source)

	remaining_write_count_list[u][ss][sp][data_halo][source] -= 1
	
	counter = remaining_write_count_list[u][ss][sp][data_halo][source]

#	print "NOT", u, ss, sp, halo, source, counter

	if counter == 0:
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

	log("rank%d, \"%s\", u=%d, writing in rank%d and counter%d"%(rank, name, u, source, counter),'general',log_type)

	launch_task(source_list=[source])
	if VIVALDI_DYNAMIC:
		launch_task()

def Free(u,ss,sp,data_halo):
	free_list = []

	# free exist u, ss, sp, halo
	# in the computing units
#	print "FREE MEM", u, ss, sp, data_halo
	not_valid = False
	rc = remaining_write_count_list[u][ss][sp]

	for execid in rc[data_halo]:
		if execid not in free_list:
			# check another buffer_halo retain_counter is remaining
			flag = True
			"""
			for other_halo in rc:
				print "OTHER", execid, halo, other_halo, rc[other_halo]
				if execid in rc[other_halo]:
					print "ELEM", execid, other_halo in retain_count[u][ss][sp]
					if other_halo in retain_count[u][ss][sp] and retain_count[u][ss][sp][other_halo] > 0:
						flag = False
						break
			"""

			for other_halo in retain_count[u][ss][sp]:
				if other_halo == data_halo: continue
				if retain_count[u][ss][sp][other_halo] > 0:
					flag = False
					break

			if flag:
				not_valid = True
				free_list.append(execid)
	
#	print "FREE list", free_list, u, ss, sp, halo, "FLAG", flag, not_valid
#	print_write_count()
#	print_retain_count()

#	if u == 1:
#		if sp == str({'y': 1, 'x': 1, 'z': 2, 'w': 1}):
#			assert(False)

	if not_valid:
		if u in valid_list:
			if ss in valid_list[u]:
				if sp in valid_list[u][ss]: 
					del valid_list[u][ss][sp]
					if valid_list[u][ss] == {}: del valid_list[u][ss]

	for dest in free_list:
		if dest == 3: continue
		if dest != None:
			tag = 0
			if isCPU(dest): tag = 4
			if isGPU(dest): tag = 5
			comm.isend(rank,		dest=dest,	tag=tag)
			comm.isend("free",	dest=dest,	tag=tag)
			tag = int("%d%d"%(tag,8))
			comm.isend(u,		dest=dest,	tag=tag)
			comm.isend(ss,		dest=dest,	tag=tag)
			comm.isend(sp,		dest=dest,	tag=tag)
			comm.isend(data_halo,dest=dest,	tag=tag)

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
#			del(depth_list[u])
#			if u in compositing_method: del(compositing_method[u])


	target = (u,ss,sp,data_halo)
	# free related to function list
	#################################################3
	for function_package in function_list:
		dp = function_package.output
		du = dp.unique_id
		dss = dp.split_shape
		dsp = dp.split_position
		dh = dp.data_halo

		if target == (du,dss,dsp,dh):
			function_list.remove(function_package)
			arg_list = function_package.function_args
			for elem in arg_list:
				au = elem.unique_id
				if au == -1:continue
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
		du = dp.unique_id
		dss = dp.split_shape
		dsp = dp.split_position
		dh = dp.data_halo

		if target == (du,dss,dsp,dh):
			data_list.remove(elem)

def release(data_package, dirty = False):
	dp = data_package

	u = dp.unique_id
	ss = str(dp.split_shape)
	sp = str(dp.split_position)
	data_halo = dp.data_halo

	try:
		retain_count[u][ss][sp][data_halo] -= 1
	except:
		print "UU", u, ss, sp, data_halo
		print retain_count
		assert(False)
	if log_type in ['retain_count']:
		log("release %d %s %s %d count:%d"%(u, ss, sp, data_halo, retain_count[u][ss][sp][data_halo]),'retain_count',log_type)

	# main lost pointer
	if u not in dirty_dict: dirty_dict[u] = dirty
	if dirty: dirty_dict[u] = True
	dirty = dirty_dict[u]

#	if retain_count[u][ss][sp][halo] == 0 and dirty:
	if retain_count[u][ss][sp][data_halo] == 0:
		if dirty:
			log("Free with dirty %d %s %s %u "%(u, ss, sp, data_halo),'retain_count',log_type)
#			print "SOURCE", source
			Free(u,ss,sp,data_halo)
		else:
			log("Free with Not dirty %d %s %s %u "%(u, ss, sp, data_halo),'retain_count',log_type)
			Free(u,ss,sp,data_halo)

def retain(data_package):
	dp = data_package

	u = dp.unique_id
	ss = dp.split_shape
	sp = dp.split_position
	data_halo = dp.data_halo
	
	if u not in retain_count: retain_count[u] = {}
	if ss not in retain_count[u]: retain_count[u][ss] = {}
	if sp not in retain_count[u][ss]: retain_count[u][ss][sp] = {}
	if data_halo not in retain_count[u][ss][sp]: retain_count[u][ss][sp][data_halo] = 0

	retain_count[u][ss][sp][data_halo] += 1
	
	if log_type in ['retain_count']:
		log("retain %d %s %s %d count:%d"%(u, ss, sp, data_halo, retain_count[u][ss][sp][data_halo]),'retain_count',log_type)

	
		
# main
##################################################################################################
if __name__ == "__main__":

	if '-L' in sys.argv:
		idx = sys.argv.index('-L')
		log_type = sys.argv[idx+1]

	VIVALDI_BLOCKING = False
	if '-B' in sys.argv:
		idx = sys.argv.index('-B')
		VIVALDI_BLOCKING = sys.argv[idx+1]
		if VIVALDI_BLOCKING.lower() in ['true','on','block','blocking']: VIVALDI_BLOCKING = True
		elif VIVALDI_BLOCKING.lower() in ['false','off','nonblock','nonblocking']: VIVALDI_BLOCKING = False

	VIVALDI_SCHEDULE = False
	if '-S' in sys.argv:
		idx = sys.argv.index('-S')
		VIVALDI_SCHEDULE = sys.argv[idx+1]
		if VIVALDI_SCHEDULE.lower() in ['local','locality','locality-aware','locality_aware']: VIVALDI_SCHEDULE = 'locality'
		elif VIVALDI_SCHEDULE.lower() in ['round','round-robin','round_lobin']: VIVALDI_SCHEDULE = 'round_robin'

	VIVALDI_DYNAMIC = 'dynamic'
	if '-D' in sys.argv:
		idx = sys.argv.index('-D')
		VIVALDI_DYNAMIC = sys.argv[idx+1]
		if VIVALDI_DYNAMIC.lower() in ['dynamic','true']: VIVALDI_DYNAMIC = True
		elif VIVALDI_DYNAMIC.lower() in ['static','false']: VIVALDI_DYNAMIC = False

	log("rank%d, Vivaldi memory manager launch at %s"%(rank, name),'general',log_type)
	computing_unit_dict = comm.recv(source=0, tag=2)

	# initialization
	###################################################################################################
	for elem in computing_unit_dict:
		idle_list.append(elem)
	
	computing_unit_list = computing_unit_dict.keys()
	idle_list.append(3) # reader
	for elem in idle_list: making_data[elem] = []

	flag = ''
	MEMORY_TIME = time.time()
	flag_times = {}
	for elem in ["depth","release","retain","synchronize","merge","notice","inform","method","function","idle","send_order","synchronize","time1","time2","time3","time4","wait"]:
		flag_times[elem] = 0

	ST = time.time()
	# main loop
	########################################################################################################
	while flag != "finish":
		#print "MEM WAIT"
		st = time.time()
		log("rank%d, memory manager Waiting"%(rank),'general',log_type)
		source = comm.recv(source=MPI.ANY_SOURCE, tag=2)
		if type(source) == str:source = int(source)
		flag = comm.recv(source=source, tag=2)
		flag_times['wait'] += time.time() - st

		log("rank%d, memory manager, source:%d flag: %s"%(rank, source, flag),'general',log_type)
		
		if flag == "depth":
			st = time.time()
			unique_id			= comm.recv(source=source, tag=24)
			ss					= comm.recv(source=source, tag=24)
			sp					= comm.recv(source=source, tag=24)
			depth				= comm.recv(source=source, tag=24)

			u = unique_id
			log("rank%d, u=%d depth=%d"%(rank, u, depth),'general',log_type)
			
			if u not in depth_list: depth_list[u] = {}
			if ss not in depth_list[u]: depth_list[u][ss] = {}
			depth_list[u][ss][sp] = depth
			flag_times[flag] += time.time()-st

		elif flag == "release":
			st = time.time()
			data_package = comm.recv(source=source, tag=27)
			dp = data_package
			dirty = False
			if source == 1:
				dirty = True
			release(dp, dirty)
			flag_times[flag] += time.time()-st

		elif flag == "retain":
			st = time.time()	
			data_package = comm.recv(source=source, tag=28)

#			print data_package.unique_id, data_package.split_position
			retain(data_package)
			flag_times[flag] += time.time()-st

		elif flag == "synchronize":
			st = time.time()
			synchronize_flag = True
			synchronize()
			flag_times[flag] += time.time()-st

		elif flag == "merge":
			st = time.time()
			data_package		= comm.recv(source=source, tag=23)
			rg					= comm.recv(source=source, tag=23)
			func_name			= comm.recv(source=source, tag=23)
			order				= comm.recv(source=source, tag=23)
			dimension			= comm.recv(source=source, tag=23)
			
			#Anukura
			front_back			= comm.recv(source=source, tag=23)
	
			dp = data_package
			u = dp.unique_id
			log("rank%d, u=%d, \"%s\", %s, merge request come, make tasks for merge"%(rank, u, func_name, order),'general',log_type)
			function_compositing(data_package, rg, func_name, order, dimension, front_back)
			#function_compositing(data_package, rg, func_name, order, dimension)
			launch_task()
			flag_times[flag] += time.time()-st

		elif flag == "notice":
			st = time.time()
			data_package	= comm.recv(source=source, tag=21)
			notice(data_package, source)
			flag_times[flag] += time.time()-st

		elif flag == "inform":
			st = time.time()
			data_package = comm.recv(source=source, tag=25)
			dest = comm.recv(source=source, tag=25)
			inform(data_package, source, dest=dest)
			flag_times[flag] += time.time()-st

		elif flag == "method":
			st = time.time()
			u	= comm.recv(source=source,	tag=26)
			m	= comm.recv(source=source,	tag=26)
			compositing_method[u] = m
			log("rank%d, u=%d method %s"%(rank, u, m),'general',log_type) 
			flag_times[flag] += time.time()-st

		elif flag == "function":
			st = time.time()
			function_package = comm.recv(source=source, tag=22)

			register_function(function_package)
			launch_task()
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

		elif flag == "save_image_out_of_core":
			data_package = comm.recv(source=source, tag=210)
			dp = data_package
			u = dp.unique_id

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
				make_a_memcpy_task(source_package, dest_package, 3, apply_halo(data_range, halo))
				n += 1

			launch_task()

			comm.isend(2, dest=3, tag=5)
			comm.isend("prepare_out_of_core_save", dest=3, tag=5)
			send_data_package(dp, dest=3, tag=502)

		elif flag == "save_image_in_core":
			data_package	= comm.recv(source=source, tag=211)
			dp = data_package
			u = dp.unique_id
			halo = dp.buffer_halo

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
			data_range = apply_halo(data_range, -data_halo+halo)
			dest_package.set_data_range(data_range)

			# full_data_range
			full_data_range = dest_package.full_data_range
			full_data_range = apply_halo(full_data_range, -data_halo+halo)
			dest_package.set_full_data_range(full_data_range)


			make_a_memcpy_task(dp, dest_package, 3, data_range)
			launch_task()

		elif flag == "gather":
			data_package = comm.recv(source=source, tag=212)
			
			dp = data_package
			u = dp.unique_id
			buffer_halo = dp.buffer_halo

			dp.split_shape = str(SPLIT_BASE)
			dp.split_position = str(SPLIT_BASE)
			data_range = dp.full_data_range

			#retain(dp) # dest retain?
			flag = register_arg(dp.copy(), idle_list+work_list.keys()) # source retain
			
			source_package = dp.copy()
			dest_package = dp.copy()
			
			buffer_halo = dest_package.buffer_halo
			data_halo = dest_package.data_halo

			# data_range 
			data_range = dest_package.data_range

			# full_data_range
			full_data_range = dest_package.full_data_range

			make_a_memcpy_task(source_package, dest_package, source, data_range)
			launch_task()

	if log_type in ['time','all']:
		print "MEMORY_MANAGER TIME", time.time() - MEMORY_TIME
		for elem in flag_times:
			print "MEM ",elem, "\t%.3f s"%(flag_times[elem])
	comm.Barrier()
	comm.Disconnect()
