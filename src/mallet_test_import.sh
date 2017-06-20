#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 24:00:00
#SBATCH --mem=8G

module load mallet/2.0.8rc3

mallet import-file --input testinput.txt --output testinput.mallet --keep-sequence --remove-stopwords --extra-stopwords /users/alee35/scratch/land-wars-devel-data/hansardpropernames.txt --print-output 
