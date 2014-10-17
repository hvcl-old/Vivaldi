import sys, os, numpy, time
from mpi4py import MPI
from Vivaldi_memory_packages import Data_package, Function_package, TEST_package

import mpi4py
# local variables

computing_unit_list = {} # mapping rank and process
		
import mpi4py
# local variables
MPI4PYPATH = os.path.abspath(os.path.dirname(mpi4py.__path__[0]))
COMMAND = []
ARGS = []
MAXPROCS = []
INFO = []

m_key = ''
# local_functions
# scheduler
##########################################################################
filename = "Vivaldi_reader.py"
unit = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
COMMAND += [sys.executable]
ARGS += [[unit]]
MAXPROCS += [1]
	

new_comm = MPI.COMM_SELF.Spawn(sys.executable, ['Vivaldi_reader.py'], 2,root=0)
comm = new_comm.Merge()

import time
print "RRRR"
dest = 2
comm.isend(0,		  	  dest=dest,    tag=5)
comm.isend("HOIHOI",	  dest=dest,	tag=6)

time.sleep(5)
#DD = TEST_package()
DD = "#####"
comm.isend(DD, dest=dest, tag=1)

import time
time.sleep(5)
