import numpy as np
import pandas as pd
import re
import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import enchant


def tag2pos(tag, returnNone=False):
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''
    sys.stdout.write('tag2pos done')


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


# Set the paths
# path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
# path_seed = '/gpfs/data/datasci/paper-m/data/seed/'
path_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
path_seed_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'

# Load the raw data to a dataframe
with open(path_local + 'membercontributions_test.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')

# Prepare the text
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
# drop some columns
text.drop(['ID', 'DATE', 'MEMBER', 'CONSTITUENCY'], axis=1, inplace=True)
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

# Get unique words
sys.stdout.write('starting countvectorizer and fittransform on whole corpus')
vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(text.SPEECH_ACT)
words = vectorizer.get_feature_names()
sys.stdout.write('done countvectorizer and fittransform')

# Spellcheck and replace incorrectly spelled words with dummy word
dictionary = enchant.Dict("en_GB")
dummy = [word for word in words if word.isalpha() is False or dictionary.check(word) is False]
notdummy = [word for word in words if word.isalpha() and dictionary.check(word)]

# Stem/lemmatize correctly spelled words_en
stemmer = SnowballStemmer('english')
lemmatizer = WordNetLemmatizer()
stems = [stemmer.stem(word) for word in notdummy]
lemmas = lemmatize_pos(notdummy)

# Create dictionary of unique words and stems/dummy word
dummy_dict = dict(zip(dummy, ['williewaiola'] * len(dummy)))
notdummy_dict = dict(zip(notdummy, stems))
unique_words_dict = dict(dummy_dict, **notdummy_dict)
unique_words = set(unique_words_dict.values())
sys.stdout.write('dictionary made')

# Convert original corpus to stemmed corpus
def replace_count_words(row, dict):
    # write metadata to file for mallet
    with open(path_local + "mc-20170727-stemmed.txt", "a") as f:
        f.write(str(row[1]) + '\t' + str(row[2]) + '\t')
    # replace unique words with dictionary values and write to file
    vectorizer = CountVectorizer()
    vec = vectorizer.fit_transform([row[3]])
    words = vectorizer.get_feature_names()
    vec_array = vec.toarray()
    for i in range(len(words)):
        with open(path_local + "mc-20170727-stemmed.txt", "a") as f:
            f.write((str(dict.get(words[i])) + ' ') * int(vec_array[:, i]))
    # write new line for each speech act
    with open(path_local + "mc-20170727-stemmed.txt", "a") as f:
        f.write('\n')
        sys.stdout.write('speech act written to file')
    # read sa from file and create sa vector
    with open(path_local + 'mc-20170727-stemmed.txt', 'r') as f:
        sa = pd.read_csv(f, sep='\t', skiprows=row.SEQ_IND, usecols=[2])
    vectorizer2 = CountVectorizer(vocabulary=unique_words)
    vec2 = vectorizer2.fit_transform(sa)
    dummy_ind = vectorizer2.vocabulary_.get('williewaiola')
    return(vec2.toarray(), dummy_ind)
    sys.stdout.write('replace_count_words done')


group = text.groupby(["BILL", "YEAR"])
doc_term_matrix = np.zeros((len(group), len(unique_words)), dtype=int)

for name, df in group:
    group_ind = int(group.indices.get(name))
    seq_ind = df.index.tolist()
    df = df.assign(SEQ_IND=seq_ind)
    df.reset_index(inplace=True)
    num_rows_sa = df.shape[0]
    debate_matrix = np.zeros((num_rows_sa, len(unique_words)), dtype=int)
    for index, row in df.iterrows():
        # write stemmed corpus to file and build debate doc-term matrix
        sa_vec, dummy_ind = replace_count_words(row, unique_words_dict)
        debate_matrix[index, ] = sa_vec
    # sum debate matrix rows to get debate vector
    debate_vec = debate_matrix.sum(axis=0)
    # build document term matrix, debate by debate
    doc_term_matrix[group_ind, ] = debate_vec
    # print debate index and debate name
    sys.stdout.write(group_ind, name)

doc_term_matrix.shape
nr_incorrectly_sp = doc_term_matrix[:, dummy_ind].sum()
nr_correctly_sp = doc_term_matrix.sum() - nr_incorrectly_sp

sys.stdout.write('Number incorrectly spelled: ' + str(nr_incorrectly_sp))
sys.stdout.write('Number correctly spelled: ' + str(nr_correctly_sp))
