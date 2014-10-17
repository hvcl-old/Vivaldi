import sys, os, numpy, time
from mpi4py import MPI
from Vivaldi_memory_packages import Data_package, Function_package, TEST_package

import mpi4py
# local variables

new_comm = MPI.COMM_SELF.Spawn(sys.executable, ['sender.py'], 2,root=0)
comm = new_comm.Merge()

import time
time.sleep(5)
