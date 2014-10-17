
from Vivaldi_memory_packages import TEST_package
from mpi4py import MPI

parent = MPI.Comm.Get_parent()
parent = parent.Merge()
comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

if rank == 1:
	comm.isend(0,		  dest=2,    tag=5)
	print "AA"
	DD = TEST_package()
	comm.isend(DD, dest=2, tag=52)
	print "BB"
	
if rank == 2:
	source = comm.recv(source=MPI.ANY_SOURCE,    tag=5)
	print "SOR", source
	
import time
time.sleep(5)