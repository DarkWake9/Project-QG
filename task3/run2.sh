#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=aefit2           ## Name of the job
#SBATCH --output=aefit2.out    ## Output file
#SBATCH --error=aefit2.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL

## Load the python interpreter
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task3
## module load conda
conda activate vibenv
## Execute the python script and pass the argument/input '90'
python task3a_dynesty2.py