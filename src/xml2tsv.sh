#!/bin/bash
#SBATCH -t 24:00:00
module load anaconda/2-4.1.1
python ~/code/philologic-hansard/hansard2tsv.py /gpfs/data/datasci/paper-m/raw/hansard_xml/*.xml > /gpfs/data/datasci/paper-m/data/speeches_dates/membercontributions-20161026.tsv
