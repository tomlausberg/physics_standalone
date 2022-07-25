#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -C gpu -A s1053 ./run_test_basic.sh radsw basic_stencils
mail -S "Daintout" toml@ethz.ch < `ls -Art | grep 'slurm'| tail -n 1`

