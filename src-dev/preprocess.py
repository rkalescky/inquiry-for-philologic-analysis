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
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# set input and output paths
path_output = '/users/alee35/scratch/land-wars-devel-data/'
path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
path_seed = '/gpfs/data/datasci/paper-m/data/seed'

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
    # compute decade
#    text['DECADE'] = (text['YEAR'].map(lambda x: int(x) - (int(x) % 10)))
    # remove non-alpha numeric characters from bill titles
    text['BILL'] = text['BILL'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', str(x)))

# create debate_id
text['DEBATE_ID'] = text['BILL'] + ' ' + text['ID']

# drop some columns
text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)

# convert integer speech acts to string
for index, row in text.iterrows():
    if type(row['SPEECH_ACT']) != str:
        text.loc[index, 'SPEECH_ACT'] = 'not text'
    row["SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')

# groupby year, decade, bill, and concatenate speech act with a space
text = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()

# append speech acts to text
with open(path_seed + 'four_corpus.txt', 'r') as f:
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

# lemmatize/stem text
# textlem = lemstem_df(text, 'stem')

# parallelize lemmatize/stem text
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

textlem.to_csv(path_output + "cleanbills-20170720.tsv",
               sep="\t", header=True, index=False, encoding='utf-8')
