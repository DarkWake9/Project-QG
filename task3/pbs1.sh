#! /bin/bash -l
#PBS -o task31.err
#PBS -e task31.err
#PBS -l nodes=1:ppn=32
#PBS -q long

cd /scratch/shantanu/icecube/Project-QG/task3

python3 ./task3a_dynesty1.py