import numpy as np
import pandas as pd
import sys
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import enchant
# import line_profiler


# @profile
def tag2pos(tag, returnNone=False):
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''
    sys.stdout.write('tag2pos done')
    sys.stdout.write('\n')


# @profile
def lemmatize_pos(x):
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


# function to build up dictionary of all unique words and replace words in corpus with stems
# @profile
def build_dict_replace_words(row, mdict):

    # get unique words in speech act
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform([row[2]])
    words = vectorizer.get_feature_names()

    # check dictionary for words and add if not present
    # check if word is already in dict
    not_cached = [word for word in words if word not in mdict]
    # check if word is dummy or not
    dictionary = enchant.Dict("en_GB")
    dummy = [word for word in not_cached if word.isalpha() is False or dictionary.check(word) is False]
    dummy_dict = dict(zip(dummy, ['williewaiola'] * len(dummy)))
    # stem or lemmatize
    not_dummy = [word for word in words if word.isalpha() and dictionary.check(word)]
    stems = [stemmer.stem(word) for word in not_dummy]
    # TOFIX: lemmas = lemmatize_pos(not_dummy)
    not_dummy_dict = dict(zip(not_dummy, stems))
    # update dictionary
    mdict.update(dummy_dict)
    mdict.update(not_dummy_dict)
    sys.stdout.write('number of keys in master dict = {}'.format(len(mdict)))
    sys.stdout.write('\n')

    # replace words with stems or dummy
    veca = vec.toarray()
    # write metadata to file for mallet
    with open(path + "mc-20170727-stemmed.txt", "a") as f:
        f.write(str(row[0]) + '\t' + str(row[1]) + '\t')
    # write speech act with stems or dummy
    for i in range(len(words)):
        with open(path + "mc-20170727-stemmed.txt", "a") as f:
            f.write((str(mdict.get(words[i])) + ' ') * int(veca[:, i]))
    # insert new line character after each speech act
    with open(path + "mc-20170727-stemmed.txt", "a") as f:
        f.write('\n')
        sys.stdout.write('speech act {} written to file'.format(index))
        sys.stdout.write('\n')


# function to count correctly spelled and incorrectly spelled words
# @profile
def count_words(row, mdict):
    # read sa from file and create sa vector
    with open(path + 'mc-20170727-stemmed.txt', 'r') as f:
        sa = pd.read_csv(f, sep='\t', skiprows=row.SEQ_IND, usecols=[2])
    vectorizer2 = CountVectorizer(vocabulary=mdict)
    vec2 = vectorizer2.fit_transform(sa)
    dummy_ind = vectorizer2.vocabulary_.get('williewaiola')
    sys.stdout.write('speech act {} added to debate {} matrix'.format(row.SEQ_IND, name))
    sys.stdout.write('\n')

    return(vec2.toarray(), dummy_ind)


# Set the paths
# path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
# path_seed = '/gpfs/data/datasci/paper-m/data/seed/'
path = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
path_seed = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'

# Load the raw data to a dataframe
# with open(path + 'membercontributions-20161026.tsv', 'r') as f:
    # text = pd.read_csv(f, sep='\t')
with open(path + 'membercontributions_test.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')
sys.stdout.write('corpus read in successfully!')
sys.stdout.write('\n')


# Prepare the text
# @profile
def prepare_text(text):
    # get year from date
    text['YEAR'] = text.DATE.str[:4]
    sys.stdout.write('get year from date!')
    sys.stdout.write('\n')
    # convert years column to numeric
    text['YEAR'] = text['YEAR'].astype(float)
    sys.stdout.write('convert years column to numeric!')
    sys.stdout.write('\n')
    # fix problems with dates and remove non-alpha numeric characters from debate titles
    for index, row in text.iterrows():
        # fix years after 1908
        if row['YEAR'] > 1908:
            text.loc[index, 'YEAR'] = np.NaN
        # # compute decade
        # text['DECADE'] = (text['YEAR'].map(lambda x: int(x) - (int(x) % 10)))
        # remove non-alpha numeric characters from bill titles
        # text.loc[index, 'BILL'] = text['BILL'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', str(x)))
        text.loc[index, 'BILL'] = str(row.BILL).translate(None, string.digits + string.punctuation)
        # convert integer speech acts to string and decode unicode strings
        if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
            text.loc[index, 'SPEECH_ACT'] = ''
        text.loc[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
    sys.stdout.write('fix problems with dates, debate titles, unicode!')
    sys.stdout.write('\n')
    # forward fill missing dates
    text['YEAR'].fillna(method='ffill', inplace=True)
    sys.stdout.write('forward fill dates!')
    sys.stdout.write('\n')
    # drop some columns
    text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)
    # write to csv
    text.to_csv(path + 'membercontributions-20170807.tsv', sep='\t', index=False)
    sys.stdout.write('processed corpus written to TSV!')
    sys.stdout.write('\n')

    # read from csv after writing once
    # with open(path + 'membercontributions-20170807.tsv', 'r') as f:
        # text = pd.read_csv(f, sep='\t')

    # groupby year, decade, bill, and concatenate speech act with a space
    text = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()
    # append seeds to text
    with open(path_seed + 'four_corpus.txt', 'r') as f:
        seed = pd.read_csv(f, sep='\t', header=None, names=['SPEECH_ACT'])
    # decode unicode string with unicode codec
    for index, row in seed.iterrows():
        seed[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
    # make metadataframe for seeds
    seed['BILL'] = ['Seed1-Napier', 'Seed2-Devon',
                    'Seed3-Richmond', 'Seed4-Bessborough']
    seed['YEAR'] = [1884, 1845, 1882, 1881]
    seed = seed[['BILL', 'YEAR', 'SPEECH_ACT']]
    # append to end of text df
    text = pd.concat([text, seed]).reset_index(drop=True)
    sys.stdout.write('corpus processed successfully!')
    sys.stdout.write('\n')

    return(text)


text = prepare_text(text)


# # get year from date
# text['YEAR'] = text.DATE.str[:4]
# sys.stdout.write('get year from date!')
# sys.stdout.write('\n')
# # convert years column to numeric
# text['YEAR'] = text['YEAR'].astype(float)
# sys.stdout.write('convert years column to numeric!')
# sys.stdout.write('\n')
# # fix problems with dates and remove non-alpha numeric characters from debate titles
# for index, row in text.iterrows():
#     # fix years after 1908
#     if row['YEAR'] > 1908:
#         text.loc[index, 'YEAR'] = np.NaN
#     # # compute decade
#     # text['DECADE'] = (text['YEAR'].map(lambda x: int(x) - (int(x) % 10)))
#     # remove non-alpha numeric characters from bill titles
#     # text.loc[index, 'BILL'] = text['BILL'].map(lambda x: re.sub(r'[^A-Za-z0-9 ]', '', str(x)))
#     text.loc[index, 'BILL'] = str(row.BILL).translate(None, string.digits + string.punctuation)
#     # convert integer speech acts to string and decode unicode strings
#     if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
#         text.loc[index, 'SPEECH_ACT'] = ''
#     text.loc[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
# sys.stdout.write('fix problems with dates, debate titles, unicode!')
# sys.stdout.write('\n')
# # forward fill missing dates
# text['YEAR'].fillna(method='ffill', inplace=True)
# sys.stdout.write('forward fill dates!')
# sys.stdout.write('\n')
# # drop some columns
# text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)
# # write to csv
# text.to_csv(path + 'membercontributions-20170807.tsv', sep='\t', index=False)
# sys.stdout.write('processed corpus written to TSV!')
# sys.stdout.write('\n')
#
# # read from csv after writing once
# # with open(path + 'membercontributions-20170807.tsv', 'r') as f:
#     # text = pd.read_csv(f, sep='\t')
#
# # groupby year, decade, bill, and concatenate speech act with a space
# text = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()
# # append seeds to text
# with open(path_seed + 'four_corpus.txt', 'r') as f:
#     seed = pd.read_csv(f, sep='\t', header=None, names=['SPEECH_ACT'])
# # decode unicode string with unicode codec
# for index, row in seed.iterrows():
#     seed[index, "SPEECH_ACT"] = row["SPEECH_ACT"].decode('utf-8')
# # make metadataframe for seeds
# seed['BILL'] = ['Seed1-Napier', 'Seed2-Devon',
#                 'Seed3-Richmond', 'Seed4-Bessborough']
# seed['YEAR'] = [1884, 1845, 1882, 1881]
# seed = seed[['BILL', 'YEAR', 'SPEECH_ACT']]
# # append to end of text df
# text = pd.concat([text, seed]).reset_index(drop=True)
# sys.stdout.write('corpus processed successfully!')
# sys.stdout.write('\n')


# Initialize a dictionary of all unique words, stemmer and lemmatizer
master_dict = {}
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()

# Write stemmed corpus to file
for index, row in text.iterrows():
    build_dict_replace_words(row, master_dict)

# Vocabulary for counting words is unique set of values in master dict
vocabulary = set(master_dict.values())
sys.stdout.write('vocabulary built successfully!')
sys.stdout.write('\n')

# Count words and build debate doc-term matrix
group = text.groupby(["BILL", "YEAR"])
doc_term_matrix = np.zeros((len(group), len(vocabulary)), dtype=int)
for name, df in group:
    group_ind = int(group.indices.get(name))
    seq_ind = df.index.tolist()
    df = df.assign(SEQ_IND=seq_ind)
    df.reset_index(inplace=True)
    num_docs = df.shape[0]
    debate_matrix = np.zeros((num_docs, len(vocabulary)), dtype=int)
    for index, row in df.iterrows():
        sa_vec, dummy_ind = count_words(row, vocabulary)
        debate_matrix[index, ] = sa_vec
    # sum debate matrix rows to get debate vector
    debate_vec = debate_matrix.sum(axis=0)
    # build document term matrix, debate by debate
    doc_term_matrix[group_ind, ] = debate_vec
sys.stdout.write('doc-term matrix built successfully!')
sys.stdout.write('\n')

# Print number of correctly/incorrectly spelled words
nr_incorrectly_sp = doc_term_matrix[:, dummy_ind].sum()
nr_correctly_sp = doc_term_matrix.sum() - nr_incorrectly_sp
sys.stdout.write('Number incorrectly spelled: ' + str(nr_incorrectly_sp))
sys.stdout.write('\n')
sys.stdout.write('Number correctly spelled: ' + str(nr_correctly_sp))
sys.stdout.write('\n')
