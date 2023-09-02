#! /bin/bash -l
#PBS -o task41.err
#PBS -e task41.err
#PBS -l nodes=1:ppn=32
#PBS -q long

cd /scratch/shantanu/icecube/Project-QG/task4

python3 ./task4a_dynesty1.py