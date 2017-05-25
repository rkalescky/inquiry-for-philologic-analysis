#!/bin/bash

#SBATCH -n 16 
#SBATCH -t 24:00:00
#SBATCH --mem=256G

module load mallet/2.0.8rc3

mallet train-topics --input /gpfs/data/datasci/alee35/land-wars-devel-data/cleanbills-20170523.mallet --num-topics 500 --optimize-interval 10 --output-state /gpfs/data/datasci/alee35/land-wars-devel-data/topic_state_500.gz --output-topic-keys /gpfs/data/datasci/alee35/land-wars-devel-data/keys_500.txt --output-doc-topics /gpfs/data/datasci/alee35/land-wars-devel-data/composition_500.txt
