#$ -cwd
#$ -N SIR_det_2stages_scriptB
#$ -o SIR_det_2stages_scriptA.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 100
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g

mpirun -np 100 python3.6 scriptB_cluster.py