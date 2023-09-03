#! /bin/bash -l
#PBS -o task42.out
#PBS -e task42.err
#PBS -l nodes=1:ppn=32
#PBS -q long

cd /scratch/shantanu/icecube/Project-QG/task4
conda activate pulsar

python3 task4a_dynesty2.py
