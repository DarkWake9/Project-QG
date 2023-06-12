#!/bin/bash                    ## SLURM JOB FOR 2nd HALF

#SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=half2      ## Name of the job
#SBATCH --output=linEbhalf2.out    ## Output file
#SBATCH --error=linEbhalf2.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=23:59:59

## Load the python interpreter
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task0
module load conda
conda activate vibenv
## Execute the python script and pass the argument/input '90'
srun python task0c_logeq_logl_mod2_linEb_half2.py