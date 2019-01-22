# inquiry-for-philologic-analysis

This code is an NLP pipeline for topic modeling large collections of documents. It is generalizeable to any text data set, once formatted properly (see `src-gen/raw_corpus2tsv.py` for an example processing script). 

This code is run on a compute cluster at Brown University using the slurm scheduler. Minor adjustments should allow the code to be run on other compute clusters or locally.

## Structure
-> data : contains tabular data output from all steps in the pipeline and two additional folders
    -> sentiment : contains positively and negatively charged adjective lists (not used in this pipeline)
    -> stoplists : contains generic and custom stopword lists
-> src : contains the code
    -> sbatch.sh : script to run the pipeline on a compute cluster in batch mode
    -> main.py : top level script with switches to control other scripts/parts in the pipeline
    -> raw_corpus2tsv.py : custom hansard script to transform raw xml data to tabular data
    -> preprocess.py : both custom hansard cleaning and generic data cleaning functions
    -> mallet_import_from_file.sh : mallet command to import data to mallet format on a compute cluster
    -> mallet_train_lda.sh : mallet command to train LDA model on mallet format data on a compute cluster
-> test : contains 10 sample Hansard xml data

## Requirements
Python 3.6.1
anaconda 3-5.2.0
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
