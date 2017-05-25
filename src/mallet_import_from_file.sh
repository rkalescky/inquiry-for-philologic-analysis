#!/bin/bash

#SBATCH -n 16 
#SBATCH -t 24:00:00
#SBATCH --mem=256G

module load mallet/2.0.8rc3

mallet import-file --input /users/alee35/scratch/land-wars-devel-data/cleanbills-20170517.tsv --output /users/alee35/scratch/land-wars-devel-data/cleanbills-20170523.mallet --keep-sequence --remove-stopwords --extra-stopwords /users/alee35/scratch/land-wars-devel-data/HansardProperNames.txt 
