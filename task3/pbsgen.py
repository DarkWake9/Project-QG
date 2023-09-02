import os

for i in range(1, 4):
    init_str = f'#! /bin/bash -l\n#PBS -o task'+ os.getcwd()[-1]+ f'{i}.err\n#PBS -e task' + os.getcwd()[-1] + f'{i}.err\n#PBS -l nodes=1:ppn=32\n#PBS -q long'
    filenames = 'task' + os.getcwd()[-1] + '_dynesty' +  str(i) + '.py'
    f = open('pbs' + str(i) + '.sh', 'w')
    
    f.write(init_str)
    f.write('\n\ncd /scratch/shantanu/icecube/Project-QG/task' + os.getcwd()[-1])
    f.write(f'\n\npython3 ./task' + os.getcwd()[-1] + 'a_dynesty' + str(i) +'.py')
    
    
    f.close()