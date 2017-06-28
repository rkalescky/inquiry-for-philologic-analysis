#!/bin/bash

#SBATCh -C intel
#SBATCH -n 16 
#SBATCH -t 48:00:00
#SBATCH --mem=256G

module load mallet/2.0.8rc3

mallet train-topics --input /users/alee35/scratch/land-wars-devel-data/cleanbills-20170626.mallet --num-topics 500 --optimize-interval 10 --output-state /users/alee35/scratch/land-wars-devel-data/topic_state_500.gz --output-topic-keys /users/alee35/scratch/land-wars-devel-data/keys_500.txt --output-doc-topics /users/alee35/scratch/land-wars-devel-data/composition_500.txt
