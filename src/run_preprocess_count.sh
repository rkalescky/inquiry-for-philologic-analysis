#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH --mem=126G
#SBATCH -t 190:00:00

python preprocess_and_count_2.py
