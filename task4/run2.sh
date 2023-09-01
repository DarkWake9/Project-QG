#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=fit2           ## Name of the job
#SBATCH --output=fit2.out    ## Output file
#SBATCH --error=fit2.err     ## Error file
#SBATCH --nodes=1              ## Number of nodes
#SBATCH --ntasks-per-node=48    ## Number of tasks per node
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL

## Load the python interpreter
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task4
## module load conda
conda activate vibenv
## Execute the python script and pass the argument/input '90'
python task4a_dynesty2.py