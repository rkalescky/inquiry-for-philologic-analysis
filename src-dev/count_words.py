import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
import enchant


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def replace_from_dict(dict, tokens):
    replaced = [str(dict.get(token, token)) for token in tokens]
    return(replaced)


# load pyenchant english dictionary
dictionary = enchant.Dict("en_GB")
# load nltk lemmatizer
lemmatizer = WordNetLemmatizer()
# load porter stemmer
stemmer = PorterStemmer()


path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
path_output = '/users/alee35/code/inquiry-for-philologic-analysis/images/'
path_local = '/Users/alee35/repos/inquiry-for-philologic-analysis/data/'
path_output_local = '/Users/alee35/repos/inquiry-for-philologic-analysis/images/'

# read the data file
# df = pd.read_csv(path_local + 'membercontributions-test.tsv',
#                  delimiter='\t', usecols=[5])
df = pd.read_csv('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/membercontributions-20181210.tsv',
                 delimiter='\t', usecols=[5])
# replace nans with empty string
df.replace(np.nan, '', regex=True, inplace=True)
# replace non string or unicode speech acts with ''
for index, row in df.iterrows():
    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
        df.loc[index, 'SPEECH_ACT'] = ''
# prepare the corpus
corpus = list(df['SPEECH_ACT'])

# ----------------------------------    

# # explore how the number of unique words change as a function of max number of
# # documents a word appears
# nr_docs = 10e0**np.linspace(0, 7, num=8)
# max_df = (nr_docs+0.5)/len(corpus)
# nr_words_all = np.zeros(len(nr_docs))
# nr_word_nonr = np.zeros(len(nr_docs))
# nr_words_en = np.zeros(len(nr_docs))
# nr_words_lemmatized = np.zeros(len(nr_docs))
# nr_words_stemmed = np.zeros(len(nr_docs))

# # get number of unique words
# for i in range(len(nr_docs)):
#     vectorizer = CountVectorizer(max_df=max_df[i])
#     vec = vectorizer.fit_transform(corpus)

#     nr_words_all[i] = np.shape(vec)[1]
#     words = vectorizer.get_feature_names()
#     # remove words with special characters and numbers in them
#     words_nonr = [word for word in words if word.isalpha()]
#     nr_word_nonr[i] = len(words_nonr)
#     # correctly spellend english words
#     words_en = [word for word in words_nonr if dictionary.check(word)]
#     nr_words_en[i] = len(words_en)
#     # lemmatize
#     lemmas = list(set([lemmatizer.lemmatize(word) for word in words_en]))
#     nr_words_lemmatized[i] = len(lemmas)
#     # stem
#     stems = list(set([stemmer.stem(word) for word in words_en]))
#     nr_words_stemmed[i] = len(stems)

#     print(nr_docs[i], nr_words_all[i], nr_word_nonr[i],
#           nr_words_en[i], nr_words_lemmatized[i], nr_words_stemmed[i])

# # cumulative words plot
# plt1, = plt.plot(nr_docs, nr_words_all)
# plt2, = plt.plot(nr_docs, nr_word_nonr)
# plt3, = plt.plot(nr_docs, nr_words_en)
# plt4, = plt.plot(nr_docs, nr_words_lemmatized)
# plt5, = plt.plot(nr_docs, nr_words_stemmed)
# plt.legend([plt1, plt2, plt3, plt4, plt5],
#            ['all words', 'words with only alpha-num. characters',
#             'correctly spelled words', 'lemmas', 'stems'],
#            loc=5, fontsize=10)

# plt.semilogx()
# plt.xlabel('max number of documents')
# plt.ylabel('number of unique words')
# plt.title('Hansard Word Frequencies')
# plt.savefig('../images/nr_words_vs_max_docs_tight.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()

# ----------------------------------    

# # explore how the total number of words change as a function of max number of
# # documents a word appears
# nr_docs = 10e0**np.linspace(0, 7, num=8)
# max_df = (nr_docs+0.5)/len(corpus)
# # total_nr_words = np.zeros(len(nr_docs))
# nr_words_alpha = np.zeros(len(nr_docs))
# nr_words_spell = np.zeros(len(nr_docs))
# nr_words_misspell = np.zeros(len(nr_docs))

# # get total number of words
# for i in range(len(nr_docs)):
#     vectorizer = CountVectorizer(max_df=max_df[i])
#     vec = vectorizer.fit_transform(corpus)
#     # total number of words in each document
#     # total_nr_words[i] = vec.toarray().sum()
#     # print(total_nr_words[i])
#     # list of all words
#     words = vectorizer.get_feature_names()
#     # remove words with special characters and numbers in them
#     words_nonr = [word for word in words if word.isalpha()]
#     alpha_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_nonr]
#     nr_words_alpha[i] = vec[:, alpha_idx].sum()
#     # correctly spelled english words
#     words_spell = [word for word in words_nonr if dictionary.check(word)]
#     spell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_spell]
#     nr_words_spell[i] = vec[:, spell_idx].sum()
#     # incorrectly spelled english words
#     words_misspell = [word for word in words_nonr if dictionary.check(word) is False]
#     misspell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_misspell]
#     nr_words_misspell[i] = vec[:, misspell_idx].sum()

#     print(nr_words_alpha[i],
#           nr_words_spell[i], nr_words_misspell[i])

# # cumulative words plot
# # plt1, = plt.plot(nr_docs, total_nr_words)
# plt2, = plt.plot(nr_docs, nr_words_alpha)
# plt3, = plt.plot(nr_docs, nr_words_spell)
# plt4, = plt.plot(nr_docs, nr_words_misspell)
# plt.legend([plt2, plt3, plt4],
#            ['words with only chars.',
#             'correctly spelled words', 'incorrectly spelled words'],
#            loc=4, fontsize=10)

# plt.semilogx()
# plt.xlabel('max # documents')
# plt.ylabel('total # words')
# plt.savefig(path_output_local + 'total_nr_words_vs_max_docs.png', dpi=150)
# plt.show()
# plt.close()


# vectorizer = CountVectorizer()
# vec = vectorizer.fit_transform(corpus)
# # total number of words in each document
# # total_nr_words = vec.toarray().sum()
# # list of all words
# words = vectorizer.get_feature_names()
# # remove words with special characters and numbers in them
# words_nonr = [word for word in words if word.isalpha()]
# # alpha_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_nonr]
# # nr_words_alpha = vec[:, alpha_idx].sum()
# # # correctly spelled english words
# # # words_spell = [word for word in words_nonr if dictionary.check(word)]
# # # spell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_spell]
# # # nr_words_spell = vec[:, spell_idx].sum()
# # # incorrectly spelled english words
# words_misspell = [word for word in words_nonr if dictionary.check(word) is False]
# # misspell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_misspell]
# # nr_words_misspell = vec[:, misspell_idx].sum()
# #
# # print('total nr. words: ' + str(len(words)))
# # print('nr. words without special characters: ' + str(len(words_nonr)))
# # print('nr. correctly spelled words: ' + str(len(words_spell)))
# # print('nr. incorrectly spelled words: ' + str(len(words_misspell)))
# # print('fr. incorrectly spelled words: ' + str(float(len(words_misspell))/len(words_nonr)))
#
# df['SPEECH_ACT'] = df['SPEECH_ACT'].astype('str').str.lower()
# df['SPEECH_ACT'] = df['SPEECH_ACT'].astype('str').str.split()
# df['SPEECH_ACT'].str.len().sum()
#
# spell_dict = dict(zip(words_misspell, [''] * len(words_misspell)))
# df['SPEECH_ACT'].apply(lambda x: replace_from_dict(spell_dict, x))
# df['SPEECH_ACT'] = df['SPEECH_ACT'].apply(' '.join)
# df['SPEECH_ACT'] = df['SPEECH_ACT'].astype('str').str.split()
# df['SPEECH_ACT'].str.len().sum()


# ----------------------------------    

# # explore number of words at each step in pre-processing
# # add dummy words to each step

# # explore how the number of unique words change as a function of max number of
# # documents a word appears
# nr_docs = np.array([10000000])
# max_df = (nr_docs+0.5)/len(corpus)

# nr_words_all = np.zeros(len(nr_docs))
# nr_words_nonr = np.zeros(len(nr_docs))
# nr_words_en = np.zeros(len(nr_docs))
# nr_words_lemmatized = np.zeros(len(nr_docs))
# nr_words_stemmed = np.zeros(len(nr_docs))

# nr_words_all_dummy = np.zeros(len(nr_docs))
# nr_words_nonr_dummy = np.zeros(len(nr_docs))
# nr_words_en_dummy = np.zeros(len(nr_docs))
# nr_words_lemmatized_dummy = np.zeros(len(nr_docs))
# nr_words_stemmed_dummy = np.zeros(len(nr_docs))

# # Read in custom stopword lists
# with open('../data/stoplists/en.txt') as f:
#     en_stop = f.read().splitlines()
# with open('../data/stoplists/stopwords-20170628.txt') as f:
#     custom_stop = f.read().splitlines()
# en_stop = [word.strip() for word in en_stop]
# custom_stop = [word.strip() for word in custom_stop]
# custom_stopwords = en_stop + custom_stop
# print('custom stopword list created successfully!')

# # get number of unique words
# for i in range(len(nr_docs)):
#     vectorizer = CountVectorizer(max_df=max_df[i])
#     vec = vectorizer.fit_transform(corpus)

#     nr_words_all[i] = np.shape(vec)[1]
#     words = vectorizer.get_feature_names()

#     words_all_dummy = [word for word in words if word in custom_stopwords]
#     nr_words_all_dummy[i] = len(words_all_dummy)

#     # remove words with special characters and numbers in them
#     words_nonr = [word for word in words if word.isalpha()]
#     nr_words_nonr[i] = len(words_nonr)

#     words_nonr_dummy = [word for word in words_nonr if word in custom_stopwords]
#     nr_words_nonr_dummy[i] = len(words_nonr_dummy)    

#     # correctly spellend english words
#     words_en = [word for word in words_nonr if dictionary.check(word)]
#     nr_words_en[i] = len(words_en)

#     words_en_dummy = [word for word in words_en if word in custom_stopwords]
#     nr_words_en_dummy[i] = len(words_en_dummy)  

#     # stem
#     stems = list(set([stemmer.stem(word) for word in words_en]))
#     nr_words_stemmed[i] = len(stems)

#     words_stemmed_dummy = [word for word in stems if word in custom_stopwords]
#     nr_words_stemmed_dummy[i] = len(words_stemmed_dummy)  

#     print('word counts')
#     print(nr_docs[i], nr_words_all[i], nr_words_nonr[i],
#           nr_words_en[i], nr_words_stemmed[i])

#     print('word counts: dummy words')
#     print(nr_docs[i], nr_words_all_dummy[i], nr_words_nonr_dummy[i],
#           nr_words_en_dummy[i], nr_words_stemmed_dummy[i])



# # count total number of words and total number of stopwords
# # with open('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/mc-stemmed20181212.txt') as f:
#     # s = pd.read_csv(file, sep='\t', usecols=[2])

# # sed "s/stopwordstop//" mc-stemmed20181212.txt > no_stopwords.txt


# ----------------------------------    
data_dt = '20181211'

# read in slightly cleaned up speech acts
with open('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/mc-midprep-' + data_dt + '.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')
print('document read in successfully!')
print(text.isnull().sum())


# ---------------------------------- 


# concatenate speech acts to full documents
deb = text.groupby(['BILL', 'YEAR'])['SPEECH_ACT'].agg(lambda x: ' '.join(x)).reset_index()
print('speech acts successfully concatenated!')
print(deb.isnull().sum())
# deb is the equivalent of the output of preprocess.prepare_custom()

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
print('custom stopword list created successfully!')
    
# # Write stemmed document to file
# for index, row in text.iterrows():
#     print(row)
#     build_dict_replace_words(row, master_dict, custom_stopwords, index)

# Pickle Master Dictionary to check topic modeling later
# with open(path + 'master_dict.pickle', 'wb') as handle:
#     pickle.dump(master_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# print('master dictionary pickled successfully!')

# Load and deserialize pickled dict
with open('../data/master_dict.pickle', 'rb') as handle:
    master_dict = pickle.load(handle)

# Vocabulary for counting words is unique set of values in master dict
vocabulary = set(master_dict.values())
print('vocabulary built successfully!')
