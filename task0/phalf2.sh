#!/bin/bash                    ## SLURM JOB FOR 1st HALF

#SBATCH --job-name=half2      ## Name of the job
#SBATCH --output=11062023half2.out    ## Output file
#SBATCH --error=11062023half2.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=1    ## Number of tasks per node
#SBATCH --cpus-per-task=40     ## Number of cores per task
#SBATCH --time=23:59:59

## Load the python interpreter
module load python

## Execute the python script and pass the argument/input '90'
srun python task0a_logeq_half1.py