#$ -cwd
#$ -N SIR_det_2stages
#$ -o SIR_det_2stages.txt
#$ -j y
#$ -S /bin/bash
#$ -pe mpi 100
#$ -l h_rt=240:00:00
#$ -l h_vmem=512g

mpirun -np 100 python3.6 SIR_det_2stages.py