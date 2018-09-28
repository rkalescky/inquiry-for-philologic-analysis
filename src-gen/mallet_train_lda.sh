#!/bin/bash

#SBATCh -C intel
#SBATCH -n 1 
#SBATCH -t 72:00:00
#SBATCH --mem=256G

module load mallet/2.0.8rc3

mallet train-topics --input /gpfs/data/datasci/paper-m/data/cleanbills-20170814.mallet --num-topics $1 --optimize-interval 10 --output-state /gpfs/data/datasci/paper-m/data/topic_state_$1.gz --output-topic-keys /gpfs/data/datasci/paper-m/data/keys_$1.txt --output-doc-topics /gpfs/data/datasci/paper-m/data/composition_$1.txt  --random-seed 999 --topic-word-weights-file /gpfs/data/datasci/paper-m/data/topic-word-weights_$1 --diagnostics-file /gpfs/data/datasci/paper-m/data/diagnostics.xml