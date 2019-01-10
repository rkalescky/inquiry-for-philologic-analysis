import os
import math
import glob2
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# set input/output paths
# path_raw = '/Users/alee35/land-wars-devel-data/02speeches_dates/'
# path_input = '/Users/alee35/land-wars-devel-data/04stemmed_bills/'
path_raw = '/Users/alee35/data/'
path_input = '/Users/alee35/repos/land-wars/ashley/'
path_output = '/Users/alee35/land-wars-devel-data/05seed2/'

# read debates metadata file
with open(path_input + 'long_bills_stemmed_metadata.tsv', 'r') as f:
    metadata = pd.read_csv(f, sep='\t', header=None)

# read raw hansard
with open(path_raw + 'membercontributions-20161026.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')
# with open(path_raw + 'membercont-test.tsv', 'r') as f:
#     text2 = pd.read_csv(f, sep='\t')
