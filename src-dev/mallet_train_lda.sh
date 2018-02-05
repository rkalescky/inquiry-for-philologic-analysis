#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH -t 72:00:00
#SBATCH --mem=256G

module load mallet/2.0.8rc3

mallet train-topics --input /gpfs/data/datasci/paper-m/data/debates/cleanbills-20170814.mallet --num-topics 500 --optimize-interval 10 --output-state /gpfs/data/datasci/paper-m/data/debates/topic_state_500.gz --output-topic-keys /gpfs/data/datasci/paper-m/data/debates/keys_500.txt --output-doc-topics /gpfs/data/datasci/paper-m/data/debates/composition_500.txt  --random-seed 999 --topic-word-weights-file /gpfs/data/datasci/paper-m/data/debates/topic-word-weights_500 --diagnostics-file /gpfs/data/datasci/paper-m/data/debates/diagnostics.xml
