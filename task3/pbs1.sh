#! /bin/bash -l
#PBS -o task31.err
#PBS -e task4tc2020623_1e5bins.err
#PBS -l nodes=1:ppn=32
#PBS -q long

#cd $PBS_O_WORKDIR
cd /scratch/shantanu/
python3 ./task4tc_sing_gamma_n2.py