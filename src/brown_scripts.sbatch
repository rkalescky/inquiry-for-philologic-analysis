#!/bin/bash
#SBATCH -J brown_scripts
#SBATCH -o brown_scripts_%j.out
#SBATCH -p standard-mem-s
#SBATCH --mem=250G

module purge
module load python mallet

python main.py

