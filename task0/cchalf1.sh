#! /bin/bash -l
#PBS -o task0ahalf1_1106.out
#PBS -e task0ahalf1_1106.err
#PBS -l nodes=1:ppn=32
#PBS -q long

#cd $PBS_O_WORKDIR
cd /scratch/shantanu/icecube/Project-QG/task0
python3 ./task0a_logeq_half1.py