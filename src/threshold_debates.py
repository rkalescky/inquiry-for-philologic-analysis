import numpy as np
import pandas as pd
import glob2


def index_array(x):
    '''
    index each column in a numpy array up to a threshold
    '''
    return np.array(x[:threshold])


def threshold_debates(percent, array, metric, year_range):
    '''
    threshold an m x n distance matrix,
    where m = main corpus documents and
          n = seed corpus documents
    '''
    # set threshold
    threshold = round(len(array) * percent)
    # get indexes of 1% most similar for each seed document vector
    sorted_array = np.argpartition(array, threshold, axis=0)
    indexed_array = np.apply_along_axis(index_array, axis=0, arr=sorted_array)
    index_set = np.unique(indexed_array.flatten())
    # get derived corpus bill titles
    dc = metadata.ix[index_set, :]
    # write to tsv
    dc.to_csv('/Users/alee35/land-wars-devel-data/05seed2/titles_derived2_{}_{}.txt'.format(metric, year_range), sep='\t', header=False, index = False)
    return(dc)


# TOFIX: get metric and year_range from filenames
def threshold_timewindows(path, percent):
    for fname in glob2.glob(path):
        print fname
        array = np.load(fname)
        metric = glob2
        year_range = glob2
        threshold_debates(percent, array, metric, year_range)


def concat_timewindows(x):
    '''
    concatenates derived corpus dataframes and
    removes duplicate rows
    '''
    path = "/Users/alee35/land-wars-devel-data/05seed2/titles_derived2_*.txt"
    for fname in glob2.glob(path):
        print(fname)


# Kullback Leibler most similar 1% debates
threshold_timewindows('/Users/alee35/land-wars-devel-data/05seed2/kld1_*.npy',
                      percent=0.01)
