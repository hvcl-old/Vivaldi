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
