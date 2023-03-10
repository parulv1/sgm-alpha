#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#### Specify job name
#$ -N cont_allstable_microint
#### Output file
#$ -o SP/$JOB_NAME_$JOB_ID.out
#### Error file
#$ -e SP/$JOB_NAME_$JOB_ID.err
#### number of cores 
#$ -pe smp 36
#### Specify queue
#$ -q long.q
#### memory per core
#$ -l mem_free=2G
#### Maximum run time 
#$ -l h_rt=150:00:00

# export PATH="/home/pverma2/software/miniconda3/bin:$PATH"
export PATH="/wynton/protected/home/rad-wynton-only/pverma2/software/miniconda3/bin:$PATH"
source activate spectrome
set -exu


export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1


nproc --all

which python 

python -u ../scripts/SGM_fit_org.py

# [[ -n "$JOB_ID" ]] && /netopt/sge_n1ge6/bin/lx24-amd64/qstat -j "$JOB_ID"
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"