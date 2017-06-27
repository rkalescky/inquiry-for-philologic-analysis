#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH -t 48:00:00
#SBATCH --mem=126G

python count_words.py
