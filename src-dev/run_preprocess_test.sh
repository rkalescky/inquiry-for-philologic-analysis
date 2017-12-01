#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 48:00:00
#SBATCH --mem=256G

python preprocess_test.py
