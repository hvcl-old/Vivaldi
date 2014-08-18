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

Add ssh-key across nodes
=======
```bash
cd ~
ssh-keygen -t rsa
chmod 700 .ssh
cd  ~/.ssh
cp id_rsa.pub authorized_keys
chmod 640 authorized_keys
```

Propagate to other nodes
=======
```bash
#!/bin/bash
for i in {1..7}
do
    ssh -t ferrari$i "cd ~/openmpi-1.8.1/build; echo password|sudo -S make all install; pwd"
done
```


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
