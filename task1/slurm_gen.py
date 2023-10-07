import os

init_str = f'#!/usr/bin/sh\n\n##SBATCH --account=vibhavasu.phy.iith'
jobtitle = 'liu_sub_xerr_fit'
g = open('slurm_'+ jobtitle +'_all.sh', 'w')
task = 'task1'
for i in range(1, 4):
    
    filenames = task + 'c_liu_sub_xerr' +  str(i) + '.py'
    
    f = open('./slurm_' + jobtitle + str(i) + '.sh', 'w')
    
    f.write(init_str)
    f.write('\n#SBATCH --job-name=' + jobtitle + str(i))
    f.write('\n#SBATCH --output=' + jobtitle + str(i) + '.out')
    f.write('\n#SBATCH --error=' + jobtitle + str(i) + '.err')
    f.write('\n#SBATCH --nodes=1\n#SBATCH --ntasks-per-node=48\n#SBATCH --time=3-23:59:59\n#SBATCH --mail-user=vibhavasu2018@gmail.com\n#SBATCH --mail-type=ALL\n## module load python')
    
    f.write('\ncd /scratch/vibhavasu.phy.iith/Project-QG/'+task+'\n## module load conda\nconda activate vibenv\nulimit -n 4096')
    
    f.write('\npython ' + filenames) 
    f.close()
    print('sbatch ' + filenames)
    g.write('sbatch slurm_' + jobtitle + str(i) + '.sh\n')
    
g.close()
