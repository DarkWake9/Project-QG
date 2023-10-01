import os

jobtitle = 'liu_sub_xerr_fit'
g = open('pbs' + jobtitle + '_all.sh', 'w')
for i in range(1, 4):
    init_str = f'#! /bin/bash -l\n#PBS -o task'+ os.getcwd()[-1]+ f'{i}.out\n#PBS -e task' + os.getcwd()[-1] + f'{i}.err\n#PBS -l nodes=1:ppn=32\n#PBS -q long'
    filenames = 'task' + os.getcwd()[-1] + 'c_liu_sub_xerr' +  str(i) + '.py'
    f = open('pbs' + str(i) + '.sh', 'w')
    
    f.write(init_str)
    f.write('\n\ncd /scratch/shantanu/icecube/Project-QG/task' + os.getcwd()[-1])
    f.write('\nconda activate pulsar')
    # f.write(f'\n\npython3 task' + os.getcwd()[-1] + 'a_dynesty' + str(i) +'.py')
    f.write(f'\n\npython3 task' + os.getcwd()[-1] + 'c_liu_sub_xerr' + str(i) +'.py')
    
    
    f.close()
    g.write('qsub ' + filenames + '\n')