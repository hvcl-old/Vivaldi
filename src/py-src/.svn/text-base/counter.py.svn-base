import time
import sys
from mpi4py import MPI
import multiprocessing

comm = MPI.Comm.Get_parent()
#parent = parent.Merge()
#comm = parent
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

#cuda.init()

CPU_count = multiprocessing.cpu_count()

comm.send(CPU_count, dest=0, tag=1)

try:
	import pycuda.driver as cuda
	import pycuda.autoinit
	GPU_count = cuda.Device.count()
	comm.send(GPU_count, dest=0, tag=2)
except:
	comm.send(0, dest=0, tag=2)

comm.Disconnect()
