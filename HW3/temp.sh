#!/bin/bash
#SBATCH -q regular
#SBATCH -C cpu
#SBATCH --output=logs/pinder.out
#SBATCH --error=logs/pinder.err
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -n 1
mamba activate ariadne2
python3 filter_pinder.py -i /pscratch/sd/r/rohans54/pinder_metadata_202402.tsv -o /pscratch/sd/r/rohans54/filter-data