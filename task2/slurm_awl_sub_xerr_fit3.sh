#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=awl_sub_xerr_fit3
#SBATCH --output=awl_sub_xerr_fit3.out
#SBATCH --error=awl_sub_xerr_fit3.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task2
## module load conda
conda activate vibenv
python task2c_awl_sub_xerr3.py