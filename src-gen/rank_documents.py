import numpy as np
import pandas as pd
import csv
from collections import Counter


def rank_docs(an_array):
    '''
    normalizes document-topic matrix by column sums, 
    and ranks documents by normalized row sums

    https://codereview.stackexchange.com/questions/65031/creating-a-list-containing-the-rank-of-the-elements-in-the-original-list
    '''
    col_sums = a.sum(axis=1)
    new_matrix = a / col_sums[:, np.newaxis]
    norm_doc_weights = new_matrix.sum(axis=0)
    indices = list(range(len(norm_doc_weights)))
    indices.sort(key=lambda x: norm_doc_weights[x])
    norm_doc_ranks = [0] * len(indices)
    for i, x in enumerate(indices): 
        norm_doc_ranks[x] = i
    # make doc rank and weight dictionary
    doc_dict = dict(zip(norm_doc_ranks, norm_doc_weights))
    return doc_dict


def subcorpus(topic_idx, n_docs):

    path = '../data/'
    with open(path + 'composition_' + n + '.txt', 'r') as f:
        comp = np.loadtxt(f, usecols=topics, delimiter='\t')
    with open(path + 'composition_' + n + '.txt', 'r') as f:
        titles = pd.read_csv(f, usecols=[1], delimiter='\t',
                            quoting=csv.QUOTE_NONE)

    subcomp = comp[:, topic_idx]

    # rank documents
    ranks_and_weights = rank_docs(subcomp)

    # filter to top n documents by normalized doc weight
    d = Counter(ranks_and_weights)
    for k, v in d.most_common(n_docs):
        print('document rank: {} has weight: {}'.format(k, v))

    # get titles of top ranked docs



