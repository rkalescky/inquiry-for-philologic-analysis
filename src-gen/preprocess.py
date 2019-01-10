#!/usr/bin/python

import numpy as np
import pandas as pd
import sys
import time
import string
import csv
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import enchant


def tag2pos(tag, returnNone=False):
    """ 
    converts part of speech tag to nltk POS object
    """
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''
    sys.stdout.write('tag2pos done')
    sys.stdout.write('\n')


def lemmatize_pos(x):
    """
    lemmatizes tokens using POS information if present
    """
    tags = pos_tag(x)
    lemmas = []
    for tag in tags:
        word = str(tag[0])
        word_tag = tag[1]
        word_pos = tag2pos(word_tag)
        if word_pos is not '':
            lemmas.append(lemmatizer.lemmatize(word, word_pos))
        else:
            lemmas.append(lemmatizer.lemmatize(word))
    return(lemmas)
    sys.stdout.write('lemmatize_pos done')
    sys.stdout.write('\n')


def prepare_text(text):
    """
    custom data cleaning and massaging for Hansard TSV
    """
    dt = time.strftime("%Y%m%d")
    path_seed = '../data/'

    # get year from date
    text['YEAR'] = text.DATE.str[:4]
    sys.stdout.write('get year from date!')
    sys.stdout.write('\n')
    # convert years column to numeric
    text['YEAR'] = text['YEAR'].astype(float)
    sys.stdout.write('convert years column to numeric!')
    sys.stdout.write('\n')
    # fix problems with dates and remove non-alpha numeric characters from document titles
    for index, row in text.iterrows():
        # fix years after 1908
        if row['YEAR'] > 1908:
            text.loc[index, 'YEAR'] = np.NaN
        # # compute decade
        # # text['DECADE'] = (text['YEAR'].map(lambda x: int(x) - (int(x) % 10)))
        # remove non-alpha numeric characters from bill titles
        trans_dict = {c: '' for c in (string.digits + string.punctuation)}
        text.loc[index, 'BILL'] = str(row.BILL).translate(str.maketrans(trans_dict))
        # convert integer speech acts to string and decode unicode strings
        # if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
        #     text.loc[index, 'SPEECH_ACT'] = str('')
        # elif type(row['SPEECH_ACT']) is unicode:
        #     text.loc[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
        # if type(row['SPEECH_ACT'] != str):
        #     text.loc[index, "SPEECH_ACT"] = str('')
    sys.stdout.write('fix problems with dates, document titles, unicode!')
    sys.stdout.write('\n')
    # filter out nan speech act rows
    text = text[pd.notnull(text['SPEECH_ACT'])]
    sys.stdout.write('drop NaN speech acts!')
    sys.stdout.write('\n')
    # forward fill missing dates
    text['YEAR'].fillna(method='ffill', inplace=True)
    text['DATE'].fillna(method='ffill', inplace=True)
    sys.stdout.write('forward fill dates!')
    sys.stdout.write('\n')
    # # concatenate BILL and DATE to get new DEBATE ID
    # text['BILL'] = text['BILL'] + ' ' + text['DATE']
    # sys.stdout.write('bill and date ID created!')
    # sys.stdout.write('\n')
    # drop some columns
    text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)
    sys.stdout.write('hansard document processed successfully!')
    sys.stdout.write('\n')

    # append seeds to text
    # path_seed = '/gpfs/data/datasci/paper-m/data/seed/'
    with open(path_seed + 'four_corpus.txt', 'r') as f:
        seed = pd.read_csv(f, sep='\t', header=None, names=['SPEECH_ACT'])
    # # decode unicode string with unicode codec
    # for index, row in seed.iterrows():
    #     seed[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
    # make metadataframe for seeds
    seed['BILL'] = ['Seed1-Napier', 'Seed2-Devon',
                    'Seed3-Richmond', 'Seed4-Bessborough']
    seed['YEAR'] = [1884, 1845, 1882, 1881]
    seed = seed[['BILL', 'YEAR', 'SPEECH_ACT']]
    # append to end of text df
    text = pd.concat([text, seed]).reset_index(drop=True)

    # remove tabs from text columns
    # remove quotes too?

    # write to csv
    text.to_csv('../data/mc-midprep-' + dt + '.tsv', sep='\t', index=False)
    sys.stdout.write('corpus and seed processed and written successfully!')
    sys.stdout.write('\n')

    return(text)


# @profile
def build_dict_replace_words(row, mdict, custom_stopwords, index):
    """
    builds up dictionary of all unique words and 
    replace words in corpus with stems
    """

    dt = time.strftime("%Y%m%d")

    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()

    # get unique words in speech act
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform([row[2]])
    words = vectorizer.get_feature_names()
    sys.stdout.write('count vectorizer and fit transform!')
    sys.stdout.write('\n')
    # check dictionary for words and add if not present
    # check if word is already in dict
    not_cached = [word for word in words if word not in mdict]
    # check if word is stopword, dummy, or not dummy
    dictionary = enchant.Dict("en_GB")
    stopword = [word for word in not_cached if word in custom_stopwords]
    stopword_dict = dict(zip(stopword, ['stopword_replacement'] * len(stopword)))
    dummy = [word for word in not_cached if word not in custom_stopwords and
             word.isalpha() is False or dictionary.check(word) is False]
    dummy_dict = dict(zip(dummy, ['substitute_word'] * len(dummy)))
    # stem or lemmatize
    not_dummy = [word for word in words if word not in custom_stopwords and
                 word.isalpha() and
                 dictionary.check(word)]
    stems = [stemmer.stem(word) for word in not_dummy]
    # TOFIX: lemmas = lemmatize_pos(not_dummy)
    not_dummy_dict = dict(zip(not_dummy, stems))
    # update dictionary
    mdict.update(stopword_dict)
    mdict.update(dummy_dict)
    mdict.update(not_dummy_dict)
    sys.stdout.write('number of keys in master dict = {}'.format(len(mdict)))
    sys.stdout.write('\n')
    # replace words with stems or dummy
    veca = vec.toarray()
    # write metadata to file for mallet
    with open("../data/mc-stemmed" + dt + ".txt", "a") as f:
        f.write(str(row[0]) + '\t' + str(row[1]) + '\t')
    # write speech act with stems or dummy
    for i in range(len(words)):
        with open("../data/mc-stemmed" + dt + ".txt", "a") as f:
            f.write((str(mdict.get(words[i])) + ' ') * int(veca[:, i]))
    # insert new line character after each speech act
    with open("../data/mc-stemmed" + dt + ".txt", "a") as f:
        f.write('\n')
        sys.stdout.write('speech act {} written to file'.format(index))
        sys.stdout.write('\n')


def read_data(file, row):
    """
    reads dataframe, reads empty dataframe if no columns
    """
    try:
        df = pd.read_csv(file, sep='\t', skiprows=row.SEQ_IND, usecols=[2],
                         quoting=csv.QUOTE_NONE)
    #except pd.io.common.EmptyDataError:
    except IOError:
        sys.stdout.write('cannot read speech act into dataframe')
        sys.stdout.write('\n')
        df = pd.DataFrame()
    return df


def count_words(row, mdict, name):
    """
    counts correctly spelled and incorrectly spelled words
    """
    # read sa from file and create sa vector
    # with open("../data/mc-stemmed" + date + ".txt", 'r') as f:
    #    sa = pd.read_csv(f, sep='\t', skiprows=row.SEQ_IND, usecols=[2],
    #                     quoting=csv.QUOTE_NONE)
    dt = time.strftime("%Y%m%d")

    sa = read_data("../data/mc-stemmed" + dt + ".txt", row)
    vectorizer2 = CountVectorizer(vocabulary=mdict)
    print(sa)
    vec2 = vectorizer2.fit_transform(sa)
    print("test2")
    if sa.shape[0] > 0:
        dummy_ind = vectorizer2.vocabulary_.get('substitute_word')
        stopword_ind = vectorizer2.vocabulary_.get('stopword_replacement')
        vec2 = vec2.toarray()
        sys.stdout.write('speech act {} added to document {} matrix'.format(row.SEQ_IND, name))
        sys.stdout.write('\n')
    else:
        dummy_ind = 0
        stopword_ind = 0
        vec2 = np.zeros((1, len(mdict)))
        sys.stdout.write('speech act {} EMPTY and zeroes added to document {} matrix'.format(row.SEQ_IND, name))
        sys.stdout.write('\n')
    return(vec2, dummy_ind, stopword_ind)


def prepare_custom(data_dt):
    """
    custom data preparation for Hansard TSV
    """
    dt = time.strftime("%Y%m%d")

    # Write to a log file
    sys.stdout = open('../logs/log-' + dt + '.txt', 'w')

    # ----------------------------------
    # Load the raw data to a dataframe
    with open('../data/membercontributions-' + data_dt + '.tsv', 'r') as f:
        text = pd.read_csv(f, sep='\t')
    sys.stdout.write('document read in successfully!')
    sys.stdout.write('\n')

    # Remove rows with missing speech acts
    text = text[pd.notnull(text.SPEECH_ACT)]
    sys.stdout.write('removed 110 rows with missing speech acts')
    sys.stdout.write('\n')

    # Prepare the Text
    text = prepare_text(text)
    

    # Read from csv after doing prepare_text once
    # with open('../data/mc-midprep-' + data_dt + '.tsv', 'r') as f:
    # text = pd.read_csv(f, sep='\t')
    # sys.stdout.write('document read in successfully!')
    # sys.stdout.write('\n')
    # print(text.isnull().sum())
    # ----------------------------------    

    # Concatenate speech acts to full documents
    deb = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()
    sys.stdout.write('speech acts successfully concatenated!')
    sys.stdout.write('\n')
    print(deb.isnull().sum())

    return deb


def prepare_data(text):
    """
    prepare the standardized TSV for MALLET and print some stats about the documents
    """
    # Initialize a dictionary of all unique words, stemmer and lemmatizer
    master_dict = {}
    path = '../data/'

    # Read in custom stopword lists
    with open('../data/stoplists/en.txt') as f:
        en_stop = f.read().splitlines()
    with open('../data/stoplists/stopwords-20170628.txt') as f:
        custom_stop = f.read().splitlines()
    en_stop = [word.strip() for word in en_stop]
    custom_stop = [word.strip() for word in custom_stop]
    custom_stopwords = en_stop + custom_stop
    sys.stdout.write('custom stopword list created successfully!')
    sys.stdout.write('\n')
     
    # Write stemmed document to file
    for index, row in text.iterrows():
        print(row)
        build_dict_replace_words(row, master_dict, custom_stopwords, index)

    # Pickle Master Dictionary to check topic modeling later
    with open(path + 'master_dict.pickle', 'wb') as handle:
        pickle.dump(master_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # Load and deserialize pickled dict
    # with open('../data/master_dict.pickle', 'rb') as handle:
    #     master_dict = pickle.load(handle)
    sys.stdout.write('master dictionary pickled successfully!')
    sys.stdout.write('\n')

    # Vocabulary for counting words is unique set of values in master dict
    vocabulary = set(master_dict.values())
    sys.stdout.write('vocabulary built successfully!')
    sys.stdout.write('\n')

    # Count words and build document doc-term matrix
    group = text.groupby(["BILL"], sort=False)
    doc_term_matrix = np.zeros((len(group), len(vocabulary)), dtype=int)

    group_ind = 0
    for name, df in group:
        # need document index, speech act index, and sa within document index
        # group_ind = group.indices.get(name)     # document index
        print("group_ind: " + str(group_ind))
        seq_ind = df.index.tolist()             # sa index
        df = df.assign(SEQ_IND=seq_ind)
        df.reset_index(inplace=True)            # sa w/i document index
        # initialize document matrix
        num_docs = df.shape[0]
        document_matrix = np.zeros((num_docs, len(vocabulary)), dtype=int)
        # fill document matrix, speech act by speech act
        for index, row in df.iterrows():
            print("===========")
            sa_vec, dummy_ind, stopword_ind = count_words(row, vocabulary, name)
            document_matrix[index, ] = sa_vec
        # sum document matrix rows to get document vector
        document_vec = document_matrix.sum(axis=0)
        # build document term matrix, document by document
        doc_term_matrix[group_ind, ] = document_vec
        # add one to document index
        group_ind += 1
    sys.stdout.write('doc-term matrix built successfully!')
    sys.stdout.write('\n')

    # Print number of correctly/incorrectly spelled words
    nr_stopwords = doc_term_matrix[:, stopword_ind].sum()
    nr_incorrectly_sp = doc_term_matrix[:, dummy_ind].sum()
    nr_correctly_sp = doc_term_matrix.sum() - nr_incorrectly_sp
    sys.stdout.write('Number stopwords: ' + str(nr_stopwords))
    sys.stdout.write('\n')
    sys.stdout.write('Number incorrectly spelled: ' + str(nr_incorrectly_sp))
    sys.stdout.write('\n')
    sys.stdout.write('Number correctly spelled: ' + str(nr_correctly_sp))
    sys.stdout.write('\n')
