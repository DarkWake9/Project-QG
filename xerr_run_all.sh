#!/usr/bin/sh

cd ./task1/
chmod +x slurm_liu_sub_xerr_fit_all.sh
sbatch slurm_liu_sub_xerr_fit_all.sh

cd ../task2/
chmod +x slurm_awl_sub_xerr_fit_all.sh
sbatch slurm_awl_sub_xerr_fit_all.sh

cd ../task3/
chmod +x slurm_awl_sup_xerr_fit_all.sh
sbatch slurm_awl_sup_xerr_fit_all.sh

cd ../task4/
chmod +x slurm_liu_sup_xerr_fit_all.sh
sbatch slurm_liu_sup_xerr_fit_all.sh
