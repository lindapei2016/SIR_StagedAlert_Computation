#$ -cwd
#$ -N SIR_det_2stages_scriptD
#$ -o SIR_det_2stages_scriptD.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 25
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g

mpirun -np 25 python3.6 scriptD_cluster.py