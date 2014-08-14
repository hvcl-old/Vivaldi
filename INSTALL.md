Download and install OpenMPI on one node
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

Download and install CUDA on one node
=======
```bash

```
Download and install CMake on one node
=======

Propagate to other nodes
=======

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
