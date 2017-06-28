#!/bin/bash

#SBATCH -C intel
#SBATCH -n 12
#SBATCH -t 72:00:00
#SBATCH --mem=126G

python preprocess.py
