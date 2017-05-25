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


def overlap_timewindows(path):
    '''
    finds overlapping debates in adjacent time windows
    '''
    years_list = []
    num_overlap = []
    num_total = []
    overlap_list = []

    timewindows = glob2.glob(path)
    pairs = ([[timewindows[i], timewindows[i + 1]]
             for i in range(len(timewindows) - 1)])

    for [current_f, next_f] in pairs:
        # get year_start from filename
        path = os.path.splitext(current_f)[0]
        path_split = os.path.basename(path).split('_')
        year_start = path_split[2]
        year_end = int(year_start) + 6

        # get overlapping debate titles
        f1 = pd.read_csv(current_f, sep='\t', header=None)
        f2 = pd.read_csv(next_f, sep='\t', header=None)
        overlap = f1.merge(f2, how='inner', on=[0, 1, 2])
        overlap.to_csv(config.path_output +
                       'DC/overlap_{}_{}.txt'.format(year_start, year_end),
                       sep='\t', header=False, index=False)

        # count overlapping and total debates
        years_list.append(year_start + ' - ' + str(year_end))
        num_overlap.append(overlap.shape[0])
        num_total.append(f1.shape[0] + f2.shape[0])
        overlap_list.append(overlap[2])

    overlap_nums = pd.DataFrame({'years': years_list,
                                 'num_overlap': num_overlap,
                                 'num_total': num_total})
    return(overlap_nums)


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
# test if there is a bug for DC2.b
threshold(config.path_output + 'kld1_*.npy', percent=.99)

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
# test if there is a bug for DC2.b
config.concat_df99 = concat_timewindows(config.path_output +
                                        'DC/debates_*_0.99.txt')

# find overlap between timewindows for 1% tranche
config.overlap_nums = overlap_timewindows(config.path_output +
                                          'DC/debates_kld1_*0.01.txt')

# concatenate overlapping debates
config.concat_overlap = concat_timewindows(config.path_output +
                                           'DC/overlap_*.txt')
