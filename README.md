# inquiry-for-philologic-analysis

This code is an NLP pipeline for topic modeling large collections of documents. It is generalizeable to any text data set, once formatted properly (see `src-gen/raw_corpus2tsv.py` for an example processing script). 

This code is run on a compute cluster at Brown University using the slurm scheduler. Minor adjustments should allow the code to be run on other compute clusters or locally.

## Requirements
Python 3.6.1
MALLET
numpy
pandas
sys
os
time
string
csv
pickle
sklearn
nltk
enchant
