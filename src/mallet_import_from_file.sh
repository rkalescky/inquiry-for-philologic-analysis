#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH -t 24:00:00
#SBATCH --mem=64G

module load mallet/2.0.8rc3

mallet import-file --input /users/alee35/scratch/land-wars-devel-data/cleanbills-20170626.tsv --output /users/alee35/scratch/land-wars-devel-data/cleanbills-20170626.mallet --keep-sequence --remove-stopwords --extra-stopwords /users/alee35/code/inquiry-for-philologic-analysis/data/stopwords-20170628.txt 
