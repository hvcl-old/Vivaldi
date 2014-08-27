Vivaldi
=======

VIsualization LAnguage for DIstributed sytstems


What is VIVALDI
=======
VIVALDI is domain specific language for heterogeneous computing system

*From Anu*

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
**A. Install Cuda Driver and toolkit**

CUDA > 5.5

**B. Install Library dependencies**

1.Openmpi > 1.7.2

2.require libraries: easy_install PIL PyOpenGL

3.PyQt4

**C. Install PyCUDA included in VIvaldi package**


**D. Install mpi4py included in Vivaldi package**


**E. add Vivaldi PATH**


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
  $ tar -zxvf Vivaldi_a_0.1.tar.gz 
  $ cd viv_alpha	
```

```bash
  $ python install.py
  $ source ~/.bashRC
```
Add node step by step to ferrari/hostfile
```bash
  $ ssh ferrari0
  $ cd hostfile
  $ vi vivaldi_machinefile
```
write below text and execute vivaldi code
In this case, ferrari0 is master node and ferrari1~7 is slave nodes.
```
ferrari0.unist.ac.kr: 
ferrari0.unist.ac.kr slots=10
-GPU=2
-G off
```
then, go to /hostfile/vivaldi_machinefile and add next node, repeatly
```
ferrari0.unist.ac.kr:
ferrari0.unist.ac.kr slots=10
-GPU=2
-G off

ferrari1.unist.ac.kr slots=10
-GPU=2
-G off
```


How to run
=======
**Vivaldi [programming code path] [options]**

*Usage: Vivaldi [file] [options]*

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



*examples*

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
*get_any_CPU()* : get any CPU available

*get_CPU_list(int n)* : get list of n CPUs  

*get_another_CPU(int e)* : get any CPU except e

# GPU functions
*get_any_GPU* : get any GPU avaiable

*get_GPU_list(int n)* : get list of n GPUs

*get_another_GPU(int e)* : get any GPU except e


# process list
*get_processor_list()* : get list of process with machine and type

*synchronize()* : wait until task que empty and every processes are idle



Usage modifiers
=======
modifier is ...
**Func(args).modifier().modifier()**


**1. range : output size, output halo can be added here for in and in&out split**

ex) func().range(x=0:123,y=-100:100,halo=..)


**2. execid : specifies execution device list**

ex) func().execid( gpu_list )



**3. split : decompose tasks using input, output or in&output method**
	output and in&output case, vivaldi automatically collect image

ex) func().split(x=2,y=2)


**4. merge : select function and order for merge input decomposition result**

*order list*

front-to-back : front data first and back data last

ex) func().merge(func_name, 'front-to-back')



**5. halo : add boundary to input decomposed data**

form:	.halo(data_name, halo_size)

ex) func().halo(image, 3)



**6. dtype : specify data type of object in the function execution**

form: 	.dtype(image, dtype).dtype(imag2, RGB)

*available dtypes :*

char :

uchar : 1 bytes unsigned integer (0 to 255)

short : 2 bytes integer (-32768 to 32767)

ushort : 2 bytes unsigned integer (0 to 65535)

int : 4 bytes integer (-2147483648 to 2147483647)

uint : 4 bytes unsigned integer (0 to 4294967296)

float : 4 bytes, single precision float, sign bit, 8 bits exponent, 23 bits mantissa

double : 8 bytes, double precision float, sign bit, 11 bits exponents, 52 bits mantissa
	
ex) func().dtype(image, short)


domain specific functions
=======
## iterators
**model view matrix is applied here**

*line_iter orthogonal_iter(T* volume, float2 p, float step)* : make perspective iterator and return line iterator travel inside the volume. 

*line_iter perspective_iter(T* volume, float x, float y, float step, float near)* : make perspective iterator and return line iterator travel inside the volume.



**model view matrix is not applied**

*line_iter(float3 from, float3 to, float d)*

float3 begin()

bool hasNext()

float3 next()

float3 direction()

*plane_iter(float2 point, float size)*

float3 begin()

bool hasNext()

float2 next()

*cube_iter(float3 point, float size)*

float3 begin()

bool hasNext()

float3 next()


##Query and Gradient functions
output of query functions is determined by input volume and image


**2D data query functions**

*point_query_2d(T* image, float2 p)* : nearest query of 2d volume

*linear_query_2d(T* image, float2 p)* : linear query of 2d volume

*linear_gradient_2d(T* image, float2 p)* : linear_gradient of 2d volume



## 3D data query functions
*point_query_3d(T* volume, float3 p)* : point query of 3d volume

*linear_query_3d(T* volume, float3 p)* : linear query of 3d volume

*cubic_query_3d(T* volume, float3 p)* :cubic query of 3d volume

*float3 linear_gradient_3d(T* volume, float3 p)* : linear gradient of 3d volume

*float3 cubic_gradient_3d(T* data, float3 p)* : cubic gradient of 3d volume



## Image processing functions

*GPU float3 phong(float3 Light_position, float3 pos, float3 N, float3 omega, float3 kd, float3 ks, float n, float3 amb)* : calculate phong shading colour using light position, normal vector and etc...

*GPU float3 diffuse(float3 Light_position, float3 N, float3 kd)* : calculate diffuse only using light position and normal vector

*template<typename R,typename T> GPU R laplacian(T* image, float2 p, VIVALDI_DATA_RANGE* sdr)* : calculate laplacian using near 4 points.


folders 
=======
**mpi4py**
It is customized for RDMA for GPU direct on PyCUDA. 
Therefore can not use ordinary mpi4py in the Vivaldi.
How to Install are in the mpi4py folder.

**PyCUDA**
PyCUDA is ordinary version.
but It is hard to install in the cluster using easy_install.
Because I added PyCUDA in the Vivaldi.

**Paper folder**
They are all related to Vivaldi paper and involving .tex and etc

**test_set**
There are Vivaldi test set we used when developing.
