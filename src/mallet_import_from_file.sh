#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH -t 24:00:00
#SBATCH --mem=64G

module load mallet/2.0.8rc3

mallet import-file --input /gpfs/data/datasci/paper-m/data/debates/mc-20170814-stemmed.txt --output /gpfs/data/datasci/paper-m/data/debates/cleanbills-20170820.mallet --keep-sequence --remove-stopwords --extra-stopwords /users/alee35/code/inquiry-for-philologic-analysis/data/stopwords-20170628.txt 
