import numpy as np
import pandas as pd
import re
import random
import multiprocessing
from functools import partial
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import enchant

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

# test
# adict = dict(zip(['Harry', 'Ron', 'Hermione'],
#                  ['Owl', 'Rat', 'Cat']))
# string = 'Harry and Ron and Hermione are magical.'
# tokens = word_tokenize(string)
# test = replace_from_dict(adict, tokens)
# test_out = ' '.join(test)


def replace_from_dict(dict, tokens):
    replaced = [str(dict.get(token, token)) for token in tokens]
    return(replaced)


def tag2pos(tag, returnNone=False):
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''


def lemstem_df(df, method):
    # lemmatize speech acts
    for index, row in df.iterrows():
        # initialize lemma list
        lemstem_list = []
        # tokenize word_pos
        if type(row['SPEECH_ACT']) == str or type(row['SPEECH_ACT']) == unicode:
            tokens = word_tokenize(row['SPEECH_ACT'])
            alpha_tokens = [token for token in tokens if token.isalpha()]
            spellchecked_tokens = replace_from_dict(spell_dict, alpha_tokens)
            tagged_tokens = pos_tag(spellchecked_tokens)
            if method == 'stem':
                lemstem_list = replace_from_dict(stem_dict, spellchecked_tokens)
            elif method == 'lemma':
                for tagged_token in tagged_tokens:
                    word = str(tagged_token[0])
                    word_pos = tagged_token[1]
                    word_pos_morphed = tag2pos(word_pos)
                    if word_pos_morphed is not '':
                        lemma = lemmatizer.lemmatize(word, word_pos_morphed)
                    else:
                        lemma = lemmatizer.lemmatize(word)
                    lemstem_list.append(lemma)

            lemstem_string = ' '.join(lemstem_list)
            df.loc[index, 'CLEAN_TEXT'] = lemstem_string
        else:
            print(str(index) + " Not string")
            df.loc[index, 'SPEECH_ACT'] = ''
            df.loc[index, 'CLEAN_TEXT'] = ''
            continue

    return(df)


# load British English spell checker
dictionary = enchant.Dict("en_GB")
# load stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# set the paths
path_output = '/users/alee35/scratch/land-wars-devel-data/'
path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
path_seed = '/gpfs/data/datasci/paper-m/data/seed/'
# path_output_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
# path_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
# path_seed_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'

# load the raw data to a dataframe
with open(path_local + 'membercontributions_test.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')

# get year from date
text['YEAR'] = text.DATE.str[:4]
# convert years column to numeric
text['YEAR'] = text['YEAR'].astype(float)
# fix problems with dates and remove non-alpha numeric characters from debate titles
for index, row in text.iterrows():
    # fix years after 1908
    if row['YEAR'] > 1908:
        text.loc[index, 'YEAR'] = np.NaN
        # forward fill missing dates
        text['YEAR'] = text['YEAR'].fillna(method='ffill')
    # # compute decade
    # text['DECADE'] = (text['YEAR'].map(lambda x: int(x) - (int(x) % 10)))
    # remove non-alpha numeric characters from bill titles
    text['BILL'] = text['BILL'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', str(x)))

# TO FIX: use this to create a metadata file for the dfr topic viewer
# create debate_id
text['DEBATE_ID'] = text['BILL'] + ' ' + text['ID']

# drop some columns
text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)
# TO FIX: is this better off the same as count_words.py?
# convert integer speech acts to string and decode unicode strings
for index, row in text.iterrows():
    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
        text.loc[index, 'SPEECH_ACT'] = ''
    row["SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
# groupby year, decade, bill, and concatenate speech act with a space
text = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()

# append seeds to text
with open(path_seed_local + 'four_corpus.txt', 'r') as f:
    seed = pd.read_csv(f, sep='\t', header=None, names=['SPEECH_ACT'])
# decode unicode string with unicode codec
for index, row in seed.iterrows():
    row["SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
# make metadataframe for seeds
seed['BILL'] = ['Seed1-Napier', 'Seed2-Devon',
                'Seed3-Richmond', 'Seed4-Bessborough']
seed['YEAR'] = [1884, 1845, 1882, 1881]
seed = seed[['BILL', 'YEAR', 'SPEECH_ACT']]
# append to end of text df
text = pd.concat([text, seed]).reset_index(drop=True)

# now that the raw data has been processed, we build up the dictionary
# prepare the corpus
corpus = list(text['SPEECH_ACT'])
nr_docs = 10e0**np.linspace(0, 7, num=8)
max_df = (nr_docs+0.5)/len(corpus)
# get unique words, remove special chars, spellcheck, lemma/stem
for i in range(len(nr_docs)):
    vectorizer = CountVectorizer(max_df=max_df[i])
    vec = vectorizer.fit_transform(corpus)
    words = vectorizer.get_feature_names()
    # remove words with special characters and numbers in them
    words_nonr = [word for word in words if word.isalpha()]
    # correctly and incorrectly spelled english words
    words_en = [word for word in words_nonr if dictionary.check(word)]
    words_nonen = [word for word in words_nonr if dictionary.check(word) == False]
    # lemmatize
    # orig_lemmas = [word for word in words_en if lemmatizer.lemmatize(word) is not None]
    # lemmas = [lemmatizer.lemmatize(word) for word in words_en]
    # stem
    orig_stems = [word for word in words_en if stemmer.stem(word) is not None]
    stems = [stemmer.stem(word) for word in words_en]
# create dictionary from lists
# lemma_dict = dict(zip(orig_lemmas, lemmas))
stem_dict = dict(zip(orig_stems, stems))
# create dictionary of misspelled words
spell_dict = dict(zip(words_nonen, ['williewaiola'] * len(words_nonen)))

# inspect random sample of misspelled words
for word in random.sample(words_nonen, 100):
    sugg = dictionary.suggest(word)
    print(word, sugg)
# count occurrence of misspelled words and descending sort by count

# find and replace lemm/stems and misspelled words in all HANSARD
# create as many processes as there are CPUs on your machine
num_processes = multiprocessing.cpu_count()
# calculate the chunk size as an integer
chunk_size = int(text.shape[0]/num_processes)
# works even if the df length is not evenly divisible by num_processes
chunks = [text.ix[text.index[i:i + chunk_size]]
          for i in range(0, text.shape[0], chunk_size)]

# create our pool with `num_processes` processes
pool = multiprocessing.Pool(processes=num_processes)
# partial with fixed second argument
lemstem_df_par = partial(lemstem_df, method='stem')
# apply our function to each chunk in the list
result = pool.map(lemstem_df_par, chunks)

# combine the results from our pool to a dataframe
textlem = pd.DataFrame().reindex_like(text)
textlem['CLEAN_TEXT'] = np.NaN
for i in range(len(result)):
    textlem.ix[result[i].index] = result[i]

for index, row in textlem.iterrows():
    row['SPEECH_ACT'] = row['SPEECH_ACT'].encode('utf-8', 'ignore').decode('utf-8', 'ignore')

textlem.to_csv(path_output_local + "cleanbills-20170718_test.tsv",
               sep="\t", header=True, index=False, encoding='utf-8')
