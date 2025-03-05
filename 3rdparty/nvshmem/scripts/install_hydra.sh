#!/bin/bash

# This script downloads hydra from a static link.
# And installs it at the user-specificed location

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: ./install_hydra.sh src_dir builddir"
    echo "    src_dir: location where hydra source will be downloaded"
    echo "    builddir: installation directory"
    exit 1
fi

srcdir=$1
builddir=$2

if test -f $builddir/bin/nvshmrun.hydra; then
    echo "hydra already installed"
    exit 0
fi

mkdir -p $srcdir
cd $srcdir
#Download hydra-4.0.2 source
wget http://www.mpich.org/static/downloads/4.0.2/hydra-4.0.2.tar.gz
gunzip hydra-4.0.2.tar.gz
tar -xvf hydra-4.0.2.tar

#Install hydra
cd hydra-4.0.2
touch aclocal.m4; 
touch Makefile.am; 
touch Makefile.in; 
touch ./mpl/aclocal.m4; 
touch ./mpl/Makefile.am; 
touch ./mpl/Makefile.in;

./configure --prefix=$builddir --enable-cuda=no --enable-nvml=no
make
make install
rm -f -- $builddir/include/mpl*
mv $builddir/bin/mpiexec.hydra $builddir/bin/nvshmrun.hydra
# create a soft link with name nvshmrun
ln -s nvshmrun.hydra $builddir/bin/nvshmrun
rm -f $builddir/bin/mpiexec $builddir/bin/mpirun

echo "Hydra binaries have been installed in $builddir/bin"
