#!/bin/bash

#SBATCh -C intel
#SBATCH -n 1 
#SBATCH -t 72:00:00
#SBATCH --mem=256G

fpath='/gpfs/data/datasci/paper-m/data'

mallet train-topics --input $fpath/cleanbills-20170824.mallet --num-topics $1 --optimize-interval 10 --output-state $fpath/topic_state_$1.gz --output-topic-keys $fpath/keys_$1.txt --output-doc-topics $fpath/composition_$1.txt  --random-seed 999 --topic-word-weights-file $fpath/topic-word-weights_$1 --diagnostics-file $fpath/diagnostics.xml