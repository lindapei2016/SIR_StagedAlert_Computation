#$ -cwd
#$ -N SIR_det_2stages_scriptA
#$ -o SIR_det_2stages_scriptA.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 320
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g

mpirun -np 320 python3.6 scriptA_cluster.py