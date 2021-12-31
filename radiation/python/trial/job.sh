#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:10:00
#SBATCH --hint=nomultithread
#SBATCH --mail-user=toml@student.ethz.ch
#SBATCH --mail-type=END
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GT4PY_BACKEND="gtc:gt:gpu"
srun -C gpu -A s1053 python cup.py
srun -C gpu -A s1053 python test.py

mail -S "Daintout" toml@ethz.ch < `ls -Art | grep 'slurm'| tail -n 1`
