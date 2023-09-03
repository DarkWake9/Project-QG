#! /bin/bash -l
#PBS -o task33.out
#PBS -e task33.err
#PBS -l nodes=1:ppn=32
#PBS -q long

cd /scratch/shantanu/icecube/Project-QG/task3
conda activate pulsar

python3 ./task3a_dynesty3.py