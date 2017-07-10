#!/bin/bash

#SBATCH -C intel
#SBATCH -n 24 
#SBATCH -t 192:00:00
#SBATCH --mem=126G

python preprocess.py
