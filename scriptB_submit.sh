#$ -cwd
#$ -N SIR_det_2stages_scriptB
#$ -o SIR_det_2stages_scriptB.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 114
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g

mpirun -np 114 python3.6 scriptB_cluster.py "C"