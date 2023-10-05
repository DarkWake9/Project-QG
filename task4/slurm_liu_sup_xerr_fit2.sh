#!/usr/bin/sh

##SBATCH --account=vibhavasu.phy.iith
#SBATCH --job-name=liu_sup_xerr_fit2
#SBATCH --output=liu_sup_xerr_fit2.out
#SBATCH --error=liu_sup_xerr_fit2.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=3-23:59:59
#SBATCH --mail-user=vibhavasu2018@gmail.com
#SBATCH --mail-type=ALL
## module load python
cd /scratch/vibhavasu.phy.iith/Project-QG/task4
## module load conda
conda activate vibenv
python task4c_liu_sup_xerr2.py

exit_code=$?

exit $exit_code