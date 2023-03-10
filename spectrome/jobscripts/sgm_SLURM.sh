#!/bin/bash
#### Specify job name
#SBATCH -J sgm_cont_microint
#### Output file
#SBATCH -o SP/"%x"_"%j".out
#### Error file
#SBATCH -e SP/"%x"_"%j".err
#### number of cores 
#SBATCH -n 36
#### Specify queue
#SBATCH --partition=long
#### --nodelist=oakland,piedmont,novato,quartzhill
#### memory per core
#SBATCH --mem=2G
#### Maximum run time 
#SBATCH --time=2-00:00:00

export PATH="/home/pverma2/software/miniconda3/bin:$PATH"
export LD_LIBRARY_PATH="/home/pverma2/software/miniconda3/lib:$LD_LIBRARY_PATH"

source activate spectrome
set -exu

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

nproc --all

which python

python -u ../scripts/SGM_fit_org.py

[[ -n "$SLURM_JOB_ID" ]] && sstat --format="JobID,AveCPU,MaxRSS,MaxPages,MaxDiskRead,MaxDiskWrite" -j "$SLURM_JOB_ID"