import numpy as np
import pandas as pd
import csv


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


def subcorpus(percent, min_freq):
    idx = [49, 62, 68, 71, 101, 110, 125, 166, 249,
        319, 348, 380, 382, 401, 406, 463]
    names = ["Congested Districts Board", "Allotments", "Rent", "Compensation",
            "Leases", "Land Reclamation", "Ordinance Survey", "Construction",
            "Land Ownership", "Testimony to Rental History",
            "Private Property Transfers", "Land Court", "Land Title Registration",
            "Housing", "Eviction", "Crofters"]

    path = '../data/'
    with open(path + 'composition_' + n + '.txt', 'r') as f:
        comp = np.loadtxt(f, usecols=idx, delimiter='\t')
    with open(path + 'composition_' + n + '.txt', 'r') as f:
        titles = pd.read_csv(f, usecols=[1], delimiter='\t',
                            quoting=csv.QUOTE_NONE)

    # rank documents
    ranked_indexes = rank_docs(comp, percent, min_freq)
    ranked_indexes.shape

    # get the debate titles by row number and write to csv
    ranked_titles = titles.filter(items=ranked_indexes, axis=0)
    ranked_titles

