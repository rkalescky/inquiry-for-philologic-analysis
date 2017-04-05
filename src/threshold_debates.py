import os
import numpy as np
import pandas as pd
import glob2
import config


def threshold(path, percent):
    '''
    thresholds time windows and writes debates to DC directory
    '''
    for f in glob2.glob(path):
        # load array
        array = np.load(f)
        # set threshold
        threshold = round(len(array) * percent)
        # get metric and year_start from filename
        f = os.path.splitext(f)[0]
        f = os.path.basename(f).split('_')
        metric = f[0]
        year = f[1]
        # get indexes of 1% most similar for each seed document vector
        sorted_array = np.argpartition(array, threshold, axis=0)
        indexed_array = np.apply_along_axis(lambda x: np.array(x[:threshold]),
                                            axis=0, arr=sorted_array)
        index_set = np.unique(indexed_array.flatten())
        # slice debate titles from full metadata dataframe
        dc = config.metadata.ix[index_set, :]
        # write to tsv
        dc.to_csv(config.path_output +
                  'DC/debates_{}_{}_{}.txt'.format(metric, year, percent),
                  sep='\t', header=False, index=False)


def concat_timewindows(path):
    '''
    concatenates derived corpus dataframes and removes duplicate rows
    '''
    # append dataframes to list
    list_dfs = []
    for f in glob2.glob(path):
        df = pd.read_csv(f, sep='\t', header=None)
        list_dfs.append(df)
    # concatenate list to one dataframe
    concat_df = pd.concat(list_dfs, ignore_index=True)
    # remove duplicates
    concat_df.drop_duplicates(inplace=True)
    return(concat_df)


# threshold tranches
threshold(config.path_output + 'kld1_*.npy', percent=0.01)
threshold(config.path_output + 'kld1_*.npy', percent=0.05)
threshold(config.path_output + 'kld1_*.npy', percent=0.10)
threshold(config.path_output + 'kld1_*.npy', percent=0.15)
threshold(config.path_output + 'kld1_*.npy', percent=0.20)
threshold(config.path_output + 'kld1_*.npy', percent=0.25)

# combine tranches and remove duplicates
config.concat_df1 = concat_timewindows(config.path_output +
                                       'DC/debates_*_0.01.txt')
config.concat_df5 = concat_timewindows(config.path_output +
                                       'DC/debates_*_0.05.txt')
config.concat_df10 = concat_timewindows(config.path_output +
                                        'DC/debates_*_0.1.txt')
config.concat_df15 = concat_timewindows(config.path_output +
                                        'DC/debates_*_0.15.txt')
config.concat_df20 = concat_timewindows(config.path_output +
                                        'DC/debates_*_0.2.txt')
config.concat_df25 = concat_timewindows(config.path_output +
                                        'DC/debates_*_0.25.txt')
