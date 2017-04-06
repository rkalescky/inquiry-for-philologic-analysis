import pandas as pd


# set input/output paths
path_speechact = '/Users/alee35/land-wars-devel-data/03stemmed_speech_acts/'
path_input = '/Users/alee35/land-wars-devel-data/04stemmed_bills/'
path_output = '/Users/alee35/land-wars-devel-data/05seed2/'

# read debates metadata file
with open(path_input + 'long_bills_stemmed_metadata.tsv', 'r') as f:
    metadata = pd.read_csv(f, sep='\t', header=None)

# read speech act metadata
with open(path_speechact + 'long_dNMF_stemmed.txt', 'r') as f:
    speechacts = pd.read_csv(f, sep='\t', header=None)
speechacts
