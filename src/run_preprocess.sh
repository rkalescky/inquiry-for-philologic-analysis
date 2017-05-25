#!/bin/bash

#SBATCH -n 16
#SBATCH -t 24:00:00
#SBATCH --mem=256G

python preprocess.py
