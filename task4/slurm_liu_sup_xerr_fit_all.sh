sbatch slurm_liu_sup_xerr_fit1.sh
sbatch slurm_liu_sup_xerr_fit2.sh
exit_code = $?
if [ $exit_code -eq 0 ];
    sbatch slurm_liu_sup_xerr_fit3.sh
    fi


