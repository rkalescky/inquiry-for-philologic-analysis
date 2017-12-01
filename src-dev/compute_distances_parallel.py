import warnings
import numpy as np
import pandas as pd
from scipy.stats import entropy
from medpy.metric import histogram
import config


def jsd(x, y):
    '''
    Jensen Shannon Divergence
    Author: jonathanfriedman
    '''
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    x = np.array(x)
    y = np.array(y)
    d1 = x*np.log2(2*x/(x+y))
    d2 = y*np.log2(2*y/(x+y))
    d1[np.isnan(d1)] = 0
    d2[np.isnan(d2)] = 0
    d = 0.5*np.sum(d1+d2)
    return d


def compute_distances(metric, outfile):
    '''
    compute a distance metric between a main corpus and seed corpus
    and write to a numpy file
    '''
    scores = np.zeros(seed_doctopics.shape[0])
    for doctopic in main_doctopics:
        scores = np.vstack((scores, np.apply_along_axis(metric,
                            1, seed_doctopics, doctopic)))
    scores = np.delete(scores, [0], axis=0)
    np.save(config.path_output + '{}'.format(outfile), scores)


def chunker(seq, size):
    '''
    function to chunk dataframe into equal size chunks, except remainder
    http://stackoverflow.com/questions/25699439/how-to-iterate-over-consecutive-chunks-of-pandas-dataframe-efficiently
    '''
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


# read document composition file (mallet output)
with open(config.path_input + 'mallet_composition_500.txt', 'r') as f:
    hs = pd.read_csv(f, sep='\t', header=None, usecols=range(2, 501))
doctopics = pd.DataFrame.as_matrix(hs)

# read in seed 2
with open(config.path_input +
          'output/knockedoutbills_kld1_markedup/knockoutkld.csv', 'r') as f:
    seed2 = pd.read_csv(f, sep=',', header=None,
                        names=['THRESHOLD', 'IX', 'YEAR', 'DECADE', 'BILL'])

# sort seed2 chronologically
seed2.sort_values('YEAR', axis=0, ascending=True, inplace=True)

# chunk 50 at a time, comparing seeds against only hansards in that year range
for i in chunker(seed2, 50):
    year_min = i.YEAR.min()
    year_max = i.YEAR.max()
    seed_idx = i.IX
    seed_doctopics = doctopics[seed_idx]
    main_idx = pd.Series(config.metadata[(config.metadata[0] >= year_min) &
                         (config.metadata[0] < year_max+1)].index)
    main_doctopics = doctopics[main_idx]
    # print config.metadata.loc[main_idx, 0].min(),config.metadata.loc[main_idx, 0].max(), config.metadata.loc[main_idx, 2]

    # calculate distance matrix for each time window and pickle
    compute_distances(entropy, 'kld1_{}_{}'.format(year_min, year_max))
