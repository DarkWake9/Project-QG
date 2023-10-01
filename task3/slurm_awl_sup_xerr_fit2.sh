#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=awl_sup_xerr_fit2
#SBATCH --output=awl_sup_xerr_fit2.out
#SBATCH --error=awl_sup_xerr_fit2.err
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task3
## module load conda
conda activate vibenv
python task3c_awl_sup_xerr2.py