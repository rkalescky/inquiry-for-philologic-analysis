import numpy as np
import pandas as pd
import csv
import config

#TODO: Fix indexing throughout

idx = [49, 62, 68, 71, 101, 110, 125, 166, 249,
       319, 348, 380, 382, 401, 406, 463]
names = ["Congested Districts Board", "Allotments", "Rent", "Compensation",
         "Leases", "Land Reclamation", "Ordinance Survey", "Construction",
         "Land Ownership", "Testimony to Rental History",
         "Private Property Transfers", "Land Court", "Land Title Registration",
         "Housing", "Eviction", "Crofters"]

path = "/Users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/"
with open(path + 'data/composition_500.txt', 'r') as f:
    # comp = np.loadtxt(f, usecols=idx, delimiter='\t')
    titles = pd.read_csv(f, usecols=[0,1], delimiter='\t', quoting=csv.QUOTE_NONE)


def rank_docs(an_array, percent, min_freq):
    threshold = round(len(comp) * percent)
    indexed_array = np.array([])
    for i in range(0, an_array.shape[1]):
        ind = np.argpartition(an_array[:, i], -threshold)[-threshold:]
        max2 = an_array[:, i][ind]
        indexed_array = np.hstack((indexed_array, max2))

    # get counts of unique document indexes
    indexed_array
    index_set = np.unique(indexed_array, return_counts=True)

    # get documents greater than some minimum frequency
    doc_indexes = index_set[0]
    counts = index_set[1]
    doc_indexes = np.array([i for (i, j) in zip(counts, doc_indexes) if i >= min_freq])
    counts = np.array([j for (i, j) in zip(counts, doc_indexes) if i >= min_freq])

    # sort the document indexes by number of times they appear
    inds = counts.argsort()
    sorted_doc_indexes = doc_indexes[inds]
    return(sorted_doc_indexes)


# set threshold to a certain number of documents
ranked_indexes = rank_docs(comp, 0.1, 16)
ranked_indexes.shape

# get the debate titles and write to csv
titles.filter(items=ranked_indexes, axis=0)
