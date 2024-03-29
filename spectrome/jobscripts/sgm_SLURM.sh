#!/bin/bash
#### Specify job name
#SBATCH -J orgSGM_pearson_10fvec_03spat_median
#### Output file
#SBATCH -o /data/rajlab1/user_data/parul/spectromeP_results/results_globalSGM/alpha_experiments/jobout/"%x"_"%j".out
#### Error file
#SBATCH -e /data/rajlab1/user_data/parul/spectromeP_results/results_globalSGM/alpha_experiments/jobout/"%x"_"%j".err
#### number of cores 
#SBATCH -n 1
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

python -u ../scripts/sgm_fit_median.py

[[ -n "$SLURM_JOB_ID" ]] && sstat --format="JobID,AveCPU,MaxRSS,MaxPages,MaxDiskRead,MaxDiskWrite" -j "$SLURM_JOB_ID"