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
