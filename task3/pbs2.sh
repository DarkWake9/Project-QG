#! /bin/bash -l
#PBS -o task32.err
#PBS -e task32.err
#PBS -l nodes=1:ppn=32
#PBS -q long

cd /scratch/shantanu/icecube/Project-QG/task3

python3 ./task3a_dynesty2.py