Vivaldi
=======

VIsualization LAnguage for DIstributed sytstems


What is VIVALDI
=======
VIVALDI is domain specific language for heterogeneous computing system

// From Anu
VIVALDI is domain specific language on hybrid computing system. It works
with code that you wrote, the code can be written by users easily. It supports
some build-in functions, it enable to load data. 
In VIVALDI, users can provide functions about their works, the functions
can be run by VIVALDI. Users set a list of devices, it is passed to main
manager. So, works are divided on target devices. Also, split modifier can
divide both input and output data. Merge function can be written and applied
to code.

How to install
=======
A. Install Cuda Driver and toolkit
   CUDA > 5.5
B. Install Library dependencies
    1.Openmpi > 1.7.2
    2.require libraries: easy_install PIL PyOpenGL
    3.PyQt4

C. Install PyCUDA included in VIvaldi package
  $cd [VIVALDI_PATH]/pycuda-2013.1.1/pycuda-2013.1.1/
  $python setup.py build
  $python setup.py install

D. Install mpi4py included in Vivaldi package
  $cd [VIVALDI_PATH]/mpi4py-1.3/mpi4py-1.3
  $python setup.py build
  $sudo python setup.py install

E. add Vivaldi PATH
  cd [VIVALDI_PATH]
  $ python install.py
  $ source ~/.bash_profile


after that you can use Vivaldi command anywhere


Add ssh-key across nodes
=======
```bash
cd ~
ssh-keygen -t rsa
chmod 700 .ssh
cd ~/.ssh
cp id_rsa.pub authorized_keys
chmod 640 authorized_keys
```

Add ssh-key to communicate with github
=======
Copy the ssh-key to clip board
```bash
cat ~/.ssh/id_rsa.pub
```
Go to https://github.com/settings/ssh , create a new SSH key and paste that content to it





Modify ~/.bash_profile
=======
```bash
vi ~/.bash_profile
```

```
# .bash_profile

# Get the aliases and functions
if [ -f ~/.bashrc ]; then
        . ~/.bashrc
fi

# User specific environment and startup programs

PATH=$PATH:$HOME/bin

PATH=$PATH:/usr/local/cuda/bin
export PATH

```
```bash
source ~/.bash_profile
```


Download and install OpenMPI
=======
```bash
wget http://www.open-mpi.org/software/ompi/v1.8/downloads/openmpi-1.8.1.tar.gz
tar -xvzf openmpi-1.8.1.tar.gz
cd openmpi-1.8.1
mkdir build
cd build
../configure --help
../configure --with-cuda
sudo make all install
```

```bash
which mpirun
/usr/local/bin/mpirun
mpirun --version
mpirun (Open MPI) 1.8.1

Report bugs to http://www.open-mpi.org/community/help/
```

```bash
#!/bin/bash
for i in {0..7}
do
    ssh -t ferrari$i "cd ~/openmpi-1.8.1/build; echo password|sudo -S make all install; pwd"
done
```

Download and install CUDA
=======
```bash

```
Download and install CMake
=======
```bash
wget http://www.cmake.org/files/v3.0/cmake-3.0.1.tar.gz
tar -xvzf cmake-3.0.1.tar.gz
cd cmake-3.0.1
mkdir build
cd build
../configure --help
../configure
sudo make all install
```

```bash
#!/bin/bash
for i in {0..7}
do
    ssh -t ferrari$i "cd ~/cmake-3.0.1/build; echo password|sudo -S make all install; pwd"
done
```



Download and install PyCUDA
=======
```bash
sudo yum install freeglut-devel
sudo yum install boost-devel boost-static
wget https://pypi.python.org/packages/source/p/pycuda/pycuda-2014.1.tar.gz
tar -xvzf pycuda-2014.1.tar.gz
cd pycuda-2014.1
./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-mt --boost-thread-libname=boost_thread-mt --no-use-shipped-boost
make -j8
sudo python setup.py install
```

```bash
#!/bin/bash
for i in {0..7}
do
    ssh -t ferrari$i "echo password|sudo -S yum install freeglut-devel"
    ssh -t ferrari$i "echo password|sudo -S yum install boost-devel boost-static"
    ssh -t ferrari$i "cd ~/pycuda-2014.1; ./configure.py --cuda-root=/usr/local/cuda --cudadrv-lib-dir=/usr/lib --boost-inc-dir=/usr/include --boost-lib-dir=/usr/lib --boost-python-libname=boost_python-mt --boost-thread-libname=boost_thread-mt --no-use-shipped-boost; echo password|sudo -S python setup.py install; pwd"
done
```
 


Download and install mpi4py
=======
```bash
wget https://pypi.python.org/packages/source/m/mpi4py/mpi4py-1.3.1.tar.gz
tar -xvzf mpi4py-1.3.1.tar.gz
cd mpi4py-1.3.1
```

```bash
vi mpi.cfg
# Open MPI on ferrari
# ----------------
[ferrari]
#mpi_dir              = /home/devel/mpi/openmpi-1.5.4
mpicc                = /usr/local/bin/mpicc
mpicxx               = /usr/local/bin/mpicxx
include_dirs         = /usr/local/include
#libraries            = mpi
library_dirs         = /usr/local/lib
runtime_library_dirs = /usr/local

```

```bash
sudo python setup.py build --mpi=ferrari
```

```bash
#!/bin/bash
for i in {0..7}
do
    ssh -t ferrari$i "cd ~/mpi4py-1.3.1; echo password|sudo python setup.py build --mpi=ferrari"
done
```


Download and install Vivaldi
=======
```bash

```


How to run
=======
Vivaldi [programming code path] [options]

Usage: Vivaldi [file] [options]
Options:
  -L
  -L time					Display time log
  -L parsing				Display parsing log
  -L general				Display general log
  -L detail					Display detail log
  -L image					save all intermediate results
  -L all					Display every log
  -L progress				Display which process is working with time progress
  -L retain_count			Display change of retain_count during execution

  -hostfile
  -hostfile hostfile_name	include machine list in clusters

  -G
  -G on						turn on GPU direct(default)
  -G off					turn off GPU direct

  -B
  -B true					blocking data transaction
  -B false					non-blocking data transaction(default), it will overlap data transaction with calculation

  -S						scheduling algorithm
  -S round_robin			round_robin scheduling
  -S locality				locality aware scheduling, minimize data transaction.(default)

  -D 
  -D dynamic				consider working or idle of execution units for scheduling(default)
  -D static					not consider working or idle of execution units for scheduling



examples
ex1) Vivaldi helloworld.vvl
hello world test

ex2) Vivaldi orthogonal.vvl -L general
print general data transfer log

ex3) Vivaldi data_list.vvl -L image
save all intermediate images during data decomposition and merge process


tutorials 
=======


main functions
=======

# pre defined constant
GIGA = float(1024*1024*1024)

MEGA = float(1024*1024)

KILO = float(1024)

AXIS = ['x','y','z','w']

SPLIT_BASE = {'x':1,'y':1,'z':1,'w':1}

modifier_list = ['execid','split','range','merge']

VIVALDI_PATH = os.environ.get('vivaldi_path')+'/'

DATA_PATH = VIVALDI_PATH + '/VIVALDI/data/'


# built in functions

# CPU functions
get_any_CPU(): get any CPU available

get_CPU_list(int n): get list of n CPUs  

get_another_CPU(int e): get any CPU except e

# GPU functions
get_any_GPU: get any GPU avaiable

get_GPU_list(int n): get list of n GPUs

get_another_GPU(int e): get any GPU except e


# process list
get_processor_list(): get list of process with machine and type

synchronize(): wait until task que empty and every processes are idle


