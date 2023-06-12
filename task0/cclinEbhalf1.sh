#! /bin/bash -l
#PBS -o linEbchalf1_1206.out
#PBS -e linEbhalf1_1206.err
#PBS -l nodes=1:ppn=32
#PBS -q long

#cd $PBS_O_WORKDIR
cd /scratch/shantanu/icecube/Project-QG/task0
python3 ./task0c_logeq_logl_mod2_linEb_half1.py