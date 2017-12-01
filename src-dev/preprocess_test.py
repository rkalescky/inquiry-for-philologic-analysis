import numpy as np
import pandas as pd
import re
from functools import partial
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import multiprocessing
import enchant

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')


def tag2pos(tag, returnNone=False):
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''


def lemmatize_pos(x):
    tags = pos_tag(x)
    lemma_list = []
    for tag in tags:
        word = str(tag[0])
        word_tag = tag[1]
        word_pos = tag2pos(word_tag)
        if word_pos is not '':
            lemma_list.append(lemmatizer.lemmatize(word, word_pos))
        else:
            lemma_list.append(lemmatizer.lemmatize(word))
    return(lemma_list)


def lemstem_df(df, method):
    # convert speech act to list of tokens
    # df['SPEECH_ACT'] = df['SPEECH_ACT'].astype('str').str.split()
    # df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(lambda x: word_tokenize(x))
    for index, row in df.iterrows():
        row['SPEECH_ACT'] = word_tokenize(row['SPEECH_ACT'])
    # filter to tokens without special characters and that are correclty spelled
    df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(lambda x: [token for token in x if token.isalpha()])
    df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(lambda x: [token for token in x if dictionary.check(token)])
    # stem or lemmatize words in token list
    if method == 'lemma':
        df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(lambda x: lemmatize_pos(x))
    elif method == 'stem':
        df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(lambda x: [stemmer.stem(token) for token in x])
    else:
        print('Unknown method. Choose "lemma" or "stem".')
    # convert stemmed token list back to single string
    df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(' '.join)

    return(df)


# TEST UNICODE ASCII ERROR
#for index, row in text.iterrows():
#    print(type(row['SPEECH_ACT']))
#    row['SPEECH_ACT'] = word_tokenize(row['SPEECH_ACT'])
#text['SPEECH_ACT'] = text['SPEECH_ACT'].apply(lambda x: word_tokenize(x))
#text['SPEECH_ACT'] = text['SPEECH_ACT'].astype('str').str.split()


# load British English spell checker
#dictionary = enchant.Dict("en_GB")
# load stemmer and lemmatizer
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# test lemmatizer and stemmer
word = ['Germans']
word2 = ['apple']
word_pos = pos_tag(word)[0][1]
word_pos_morphed = tag2pos(word_pos)
word_pos2 = pos_tag(word2)[0][1]
word_pos_morphed2 = tag2pos(word_pos2)
lemmatizer.lemmatize(str(word), word_pos_morphed)
lemmatizer.lemmatize(str(word2), word_pos_morphed2)
lemmatizer.lemmatize('Germany', 'n')
lemmatizer.lemmatize('Germanic', 'a')
lemmatizer.lemmatize('German', 'n')
lemmatizer.lemmatize('Germans')
lemmatizer.lemmatize('bicycle')
lemmatizer.lemmatize('bicycles')
lemmatizer.lemmatize('bikes')
lemmatizer.lemmatize('bicycling')
stemmer.stem('German')
stemmer.stem('Germans')
stemmer.stem('Germanic')
stemmer.stem('Germany')
stemmer.stem('bicycle')
stemmer.stem('bicycles')
stemmer.stem('bikes')
stemmer.stem('bicycling')

path_output = '/users/alee35/scratch/land-wars-devel-data/'
path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
path_seed = '/gpfs/data/datasci/paper-m/data/seed/'

#path_output_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
#path_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
#path_seed_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'

with open(path + 'membercontributions-20161026.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')

# get year from date
text['YEAR'] = text.DATE.str[:4]
# convert years column to numeric
text['YEAR'] = text['YEAR'].astype(float)
# process dates
for index, row in text.iterrows():
    # fix years after 1908
    if row['YEAR'] > 1908:
        text.loc[index, 'YEAR'] = np.NaN
        # forward fill missing dates
        text['YEAR'] = text['YEAR'].fillna(method='ffill')
    # remove non-alpha numeric characters from bill titles
    text['BILL'] = text['BILL'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', str(x)))

# create debate_id
text['DEBATE_ID'] = text['BILL'] + ' ' + text['ID']
metadata = text.drop(['SPEECH_ACT'], axis=1, inplace=False)
metadata.to_csv(path_output + "metadata_20170724.tsv",
               sep="\t", header=True, index=False, encoding='utf-8')

# drop some columns
#text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)

# convert integer speech acts to string and decode unicode strings
#for index, row in text.iterrows():
#    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
#        row["SPEECH_ACT"] = row['SPEECH_ACT'].encode('utf-8', 'ignore').decode('utf-8', 'ignore')
#        text.loc[index, 'SPEECH_ACT'] = ''

# groupby year, decade, bill, and concatenate speech act with a space
#text = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()

# append speech acts to text
#with open(path_seed_local + 'four_corpus.txt', 'r') as f:
#    seed = pd.read_csv(f, sep='\t', header=None, names=['SPEECH_ACT'])

# decode unicode string with unicode codec
#for index, row in seed.iterrows():
#    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
#        row["SPEECH_ACT"] = row['SPEECH_ACT'].encode('utf-8', 'ignore').decode('utf-8', 'ignore')
#        text.loc[index, 'SPEECH_ACT'] = ''

# make metadataframe for seeds
#seed['BILL'] = ['Seed1-Napier', 'Seed2-Devon',
#                'Seed3-Richmond', 'Seed4-Bessborough']
#seed['YEAR'] = [1884, 1845, 1882, 1881]
#seed = seed[['BILL', 'YEAR', 'SPEECH_ACT']]

# append to end of text df
#text = pd.concat([text, seed]).reset_index(drop=True)


# lemmatize/stem text
# textlem = lemstem_df(text, 'lemma')

# create as many processes as there are CPUs on your machine
#num_processes = multiprocessing.cpu_count()/2
# calculate the chunk size as an integer
#chunk_size = int(text.shape[0]/num_processes)
# works even if the df length is not evenly divisible by num_processes
#chunks = [text.ix[text.index[i:i + chunk_size]]
#          for i in range(0, text.shape[0], chunk_size)]

# create our pool with `num_processes` processes
#pool = multiprocessing.Pool(processes=num_processes)
# partial with fixed second argument
#lemstem_df_par = partial(lemstem_df, method='stem')
# apply our function to each chunk in the list
#result = pool.map(lemstem_df_par, chunks)

# combine the results from our pool to a dataframe
#textlem = pd.DataFrame().reindex_like(text)
# textlem['CLEAN_TEXT'] = np.NaN
#for i in range(len(result)):
#    textlem.ix[result[i].index] = result[i]

#textlem.to_csv(path_output_local + "cleanbills-20170724_test.tsv",
#               sep="\t", header=True, index=False, encoding='utf-8')
