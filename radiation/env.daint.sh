#!/bin/bash

# This script contains functions for setting up machine specific compile
# environments for the dycore and the Fortran parts. Namely, the following
# functions must be defined in this file:

module load daint-gpu
module swap PrgEnv-cray PrgEnv-gnu
module load cray-python/3.8.2.1
module load cray-mpich/7.7.18
module load Boost/1.78.0-CrayGNU-21.09-python3
module load cudatoolkit/11.2.0_3.39-2.1__gf93aa1c
module load graphviz/2.44.0
module load /project/s1053/install/modulefiles/gcloud/303.0.0
module load cray-hdf5

module switch gcc gcc/9.3.0
module switch cray-python cray-python/3.8.2.1

# load gridtools modules
module load /project/s1053/install/modulefiles/gridtools/1_1_3
module load /project/s1053/install/modulefiles/gridtools/2_1_0

NVCC_PATH=$(which nvcc)
export CUDA_HOME=$(echo $NVCC_PATH | sed -e "s/\/bin\/nvcc//g")
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Setup RDMA for GPU. Set PIPE size to 256 (# of messages allowed in flight)
# Turn as-soon-as-possible transfer (NEMESIS_ASYNC_PROGRESS) on
export MPICH_RDMA_ENABLED_CUDA=1
export CRAY_CUDA_MPS=1
export CRAY_CUDA_PROXY=0
export MPICH_G2G_PIPELINE=256
export MPICH_NEMESIS_ASYNC_PROGRESS=1
export MPICH_MAX_THREAD_SAFETY=multiple

# the eve toolchain requires a clang-format executable, we point it to the right place
export CLANG_FORMAT_EXECUTABLE=/project/s1053/install/venv/vcm_1.0/bin/clang-format

export PYTHONPATH=/project/s1053/install/serialbox/gnu/python:$PYTHONPATH
export WHEEL_DIR=/project/s1053/install/wheeldir
