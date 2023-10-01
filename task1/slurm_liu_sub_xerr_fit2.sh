#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=liu_sub_xerr_fit2
#SBATCH --output=liu_sub_xerr_fit2.out
#SBATCH --error=liu_sub_xerr_fit2.err
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task1
## module load conda
conda activate vibenv
python task1c_liu_sub_xerr2.py