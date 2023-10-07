#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=liu_sup_xerr_fit3
#SBATCH --output=liu_sup_xerr_fit3.out
#SBATCH --error=liu_sup_xerr_fit3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task4
## module load conda
conda activate vibenv
ulimit -n 4096
python task4c_liu_sup_xerr3.py