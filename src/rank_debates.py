import numpy as np
import pandas as pd
import csv

idx = [49, 62, 68, 71, 101, 110, 125, 166, 249,
       319, 348, 380, 382, 401, 406, 463]
names = ["Congested Districts Board", "Allotments", "Rent", "Compensation",
         "Leases", "Land Reclamation", "Ordinance Survey", "Construction",
         "Land Ownership", "Testimony to Rental History",
         "Private Property Transfers", "Land Court", "Land Title Registration",
         "Housing", "Eviction", "Crofters"]

path = "/Users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/"
with open(path + 'data/composition_500.txt', 'r') as f:
    comp = np.loadtxt(f, usecols=idx, delimiter='\t')
with open(path + 'data/composition_500.txt', 'r') as f:
    titles = pd.read_csv(f, usecols=[1], delimiter='\t',
                         quoting=csv.QUOTE_NONE)


def rank_docs(an_array, percent, min_freq):

    '''
    takes an array as input and gets the row indexes for the top x percent
    values above some threshold, min_freq, for each column in the array.
    the output is an array containing the unique indexes ranked by frequency.
    '''
    threshold = round(len(an_array) * percent)
    ind_array = np.array([])
    vals_array = np.array([])

    for i in range(0, an_array.shape[1]):
        ind = np.argpartition(an_array[:, i], -threshold)[-threshold:]
        vals = an_array[:, i][ind]
        ind_array = np.hstack((ind_array, ind))
        vals_array = np.hstack((vals_array, vals))

    # get counts of unique document indexes
    ind_set = np.unique(ind_array, return_counts=True)

    # get documents greater than some minimum frequency
    ind_unique = ind_set[0]
    ind_counts = ind_set[1]

    # get document indices
    indexes = np.array([i for (i, j) in zip(ind_unique,
                                            ind_counts) if j >= min_freq])
    counts = np.array([j for (i, j) in zip(ind_unique,
                                           ind_counts) if j >= min_freq])

    # sort the document indexes by number of times they appear
    sort_order = counts[::1].argsort()
    sorted_inds = indexes[sort_order]
    return(sorted_inds)


# rank documents
ranked_indexes = rank_docs(comp, 0.1, 12)
ranked_indexes.shape

# get the debate titles by row number and write to csv
ranked_titles = titles.filter(items=ranked_indexes, axis=0)
ranked_titles

# tests
an_array = np.array([[0, 0, 1, 2, 3],
                     [2, 2, 2, 3, 4],
                     [1, 1, 9, 1, 9]])
percent = 0.1
min_freq = 2

# threshold = round(len(comp) * percent)
threshold = 1
ind_array = np.array([])
vals_array = np.array([])

for i in range(0, an_array.shape[1]):
    ind = np.argpartition(an_array[:, i], -threshold)[-threshold:]
    vals = an_array[:, i][ind]
    ind_array = np.hstack((ind_array, ind))
    vals_array = np.hstack((vals_array, vals))

# get counts of unique document indexes
ind_array
ind_set = np.unique(ind_array, return_counts=True)
vals_array
vals_set = np.unique(vals_array, return_counts=True)

# get documents greater than some minimum frequency
ind_unique = ind_set[0]
ind_counts = ind_set[1]
vals_probs = vals_set[0]
vals_counts = vals_set[1]
# get document indices
indexes = np.array([i for (i, j) in zip(ind_unique, ind_counts)
                    if j >= min_freq])
counts = np.array([j for (i, j) in zip(ind_unique, ind_counts)
                   if j >= min_freq])
# indexes = np.array([i for (i, j) in enumerate(ind_counts) if j >= min_freq])
# get document probabilities
# probs = np.array([j for (i, j) in zip(vals_counts, vals_probs)
#                   if i >= min_freq])
# counts = np.array([i for (i, j) in zip(vals_counts, vals_probs)
#                    if i >= min_freq])

# sort the document indexes by number of times they appear
sort_order = counts[::1].argsort()
sorted_inds = indexes[sort_order]
sorted_inds
