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


# read document composition file (mallet output)
with open('/Users/alee35/Dropbox (Brown)/hansard-mallet/mallet_composition_500.txt', 'r') as f:
# with open(config.path_input + 'mallet_composition_500.txt', 'r') as f:
    hs = pd.read_csv(f, sep='\t', header=None, usecols=range(2, 501))
doctopics = pd.DataFrame.as_matrix(hs)

# read in seed 2
with open(config.path_input +
          'output/knockedoutbills_kld1_markedup/knockoutkld.csv', 'r') as f:
    seed2 = pd.read_csv(f, sep=',', header=None,
                        names=['THRESHOLD', 'IX', 'YEAR', 'DECADE', 'BILL'])

# iterate through 5 year time windows, with 2 year gap between windows
# other rows become main corpus
for year in range(1800, 1904, 2):
    twindow = seed2[(seed2.YEAR >= year) & (seed2.YEAR < year+5)]
    if len(twindow) == 0:
        continue
    seed_doctopics = doctopics[twindow.IX.values]
    main_doctopics = np.delete(doctopics, twindow.IX.values, axis=0)

    # calculate distance matrix for each time window and pickle
    compute_distances(entropy, 'kld1_{}'.format(year))
    compute_distances(histogram.kullback_leibler, 'kld2_{}'.format(year))
    compute_distances(jsd, 'jsd_{}'.format(year))
