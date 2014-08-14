from mpi4py import MPI
from Vivaldi_misc import *

# init mpi
parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = parent.Get_size()
rank = parent.Get_rank()
name = MPI.Get_processor_name()


# main
##################################################################################################
if __name__ == "__main__":
	if '-L' in sys.argv:
		idx = sys.argv.index('-L')
		log_type = sys.argv[idx+1]

	log("rank%d, scheduler launch at %s"%(rank,name),'general',log_type)
	computing_unit_list = parent.recv(source=0, tag=3)

	flag = ''
	while flag != "finish":
		log("rank%d, scheduler is Waiting"%(rank),'general',log_type)
		source = parent.recv(source=MPI.ANY_SOURCE, tag=3)
		if type(source) == str:source = int(source)
		flag = parent.recv(source=source, tag=3)

	parent.Barrier()
	parent.Disconnect()
