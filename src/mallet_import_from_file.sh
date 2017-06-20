#!/bin/bash

#SBATCH -n 1 
#SBATCH -t 24:00:00
#SBATCH --mem=64G

module load mallet/2.0.8rc3

mallet import-file --input /users/alee35/scratch/land-wars-devel-data/cleanbills-20170517.tsv --output /users/alee35/scratch/land-wars-devel-data/cleanbills-20170607.mallet --keep-sequence --remove-stopwords --extra-stopwords /users/alee35/scratch/land-wars-devel-data/hansardpropernames.txt --print-output 
