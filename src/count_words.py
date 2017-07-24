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


# load pyenchant english dictionary
dictionary = enchant.Dict("en_GB")
# load nltk lemmatizer
lemmatizer = WordNetLemmatizer()
# load porter stemmer
stemmer = PorterStemmer()


path = '/gpfs/data/datasci/paper-m/data/speeches_dates/'
path_output = '/users/alee35/code/inquiry-for-philologic-analysis/images/'
# path_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/data/'
# path_output_local = '/users/alee35/Google Drive/repos/inquiry-for-philologic-analysis/images/'

# read the data file
df = pd.read_csv(path_local + 'membercontributions-20161026.tsv',
                 delimiter='\t', usecols=[5])
# replace nans with empty string
df.replace(np.nan, '', regex=True, inplace=True)
# replace non string or unicode speech acts with ''
for index, row in df.iterrows():
    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
        df.loc[index, 'SPEECH_ACT'] = ''
# prepare the corpus
corpus = list(df['SPEECH_ACT'])

# # explore how the number of unique words change as a function of max number of
# # documents a word appears
# nr_docs = 10e0**np.linspace(0, 7, num=8)
# max_df = (nr_docs+0.5)/len(corpus)
# nr_words_all = np.zeros(len(nr_docs))
# nr_word_nonr = np.zeros(len(nr_docs))
# nr_words_en = np.zeros(len(nr_docs))
# nr_words_lemmatized = np.zeros(len(nr_docs))
# nr_words_stemmed = np.zeros(len(nr_docs))
#
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
#
#     print(nr_docs[i], nr_words_all[i], nr_word_nonr[i],
#           nr_words_en[i], nr_words_lemmatized[i], nr_words_stemmed[i])
#
# # cumulative words plot
# plt1, = plt.plot(nr_docs, nr_words_all)
# plt2, = plt.plot(nr_docs, nr_word_nonr)
# plt3, = plt.plot(nr_docs, nr_words_en)
# plt4, = plt.plot(nr_docs, nr_words_lemmatized)
# plt5, = plt.plot(nr_docs, nr_words_stemmed)
# plt.legend([plt1, plt2, plt3, plt4, plt5],
#            ['all words', 'words with only chars.',
#             'correctly spelled words', 'lemmas', 'stems'],
#            loc=4, fontsize=10)
#
# plt.semilogx()
# plt.xlabel('max # documents')
# plt.ylabel('# unique words')
# plt.savefig('../images/nr_words_vs_max_docs.png', dpi=150)
# plt.show()
# plt.close()


# # explore how the total number of words change as a function of max number of
# # documents a word appears
# nr_docs = 10e0**np.linspace(0, 7, num=8)
# max_df = (nr_docs+0.5)/len(corpus)
# total_nr_words = np.zeros(len(nr_docs))
# nr_words_alpha = np.zeros(len(nr_docs))
# nr_words_spell = np.zeros(len(nr_docs))
# nr_words_misspell = np.zeros(len(nr_docs))

# # get total number of words
# for i in range(len(nr_docs)):
#     vectorizer = CountVectorizer(max_df=max_df[i])
#     vec = vectorizer.fit_transform(corpus)
#     # total number of words in each document
#     total_nr_words[i] = vec.toarray().sum()
#     print(total_nr_words[i])
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

#     print(nr_docs[i], total_nr_words[i], nr_words_alpha[i],
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


vectorizer = CountVectorizer()
vec = vectorizer.fit_transform(corpus)
# total number of words in each document
total_nr_words = vec.toarray().sum()
print(total_nr_words)
# list of all words
words = vectorizer.get_feature_names()
# remove words with special characters and numbers in them
words_nonr = [word for word in words if word.isalpha()]
alpha_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_nonr]
nr_words_alpha = vec[:, alpha_idx].sum()
# correctly spelled english words
words_spell = [word for word in words_nonr if dictionary.check(word)]
spell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_spell]
nr_words_spell = vec[:, spell_idx].sum()
# incorrectly spelled english words
words_misspell = [word for word in words_nonr if dictionary.check(word) is False]
misspell_idx = [vectorizer.vocabulary_.get(word).astype('int64') for word in words_misspell]
nr_words_misspell = vec[:, misspell_idx].sum()

print('total nr. words: ' + str(len(words)))
print('nr. words without special characters: ' + str(len(words_nonr)))
print('nr. correctly spelled words: ' + str(len(words_spell)))
print('nr. incorrectly spelled words: ' + str(len(words_misspell)))
print('fr. incorrectly spelled words: ' + str(float(len(words_misspell))/len(words_nonr)))
