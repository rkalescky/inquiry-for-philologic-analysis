#!/bin/bash

#SBATCH -C intel
#SBATCH -n 1 
#SBATCH -t 24:00:00
#SBATCH --mem=64G

fpath='/gpfs/data/datasci/paper-m/data'

mallet import-file --input $fpath/mc-20170824-stemmed.txt --output $fpath/cleanbills-20170824.mallet --keep-sequence --remove-stopwords --extra-stopwords ../data/stoplists/stopwords-20170628.txt