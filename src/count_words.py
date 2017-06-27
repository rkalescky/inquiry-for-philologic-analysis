import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
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


path = '/gpfs/data/datasci/paper-m/HANSARD/speeches_dates/'

# read the data file
df = pd.read_csv(path + 'membercontributions-20161026.tsv',
                 delimiter='\t', usecols=[5])
# replace nans with empty string
df.replace(np.nan, '', regex=True, inplace=True)
# replace non string or unicode speech acts with ''
for index, row in df.iterrows():
    if type(row['SPEECH_ACT']) != str and type(row['SPEECH_ACT']) != unicode:
        df.loc[index, 'SPEECH+_ACT'] = ''
# prepare the corpus
corpus = list(df['SPEECH_ACT'])

# explore how the number of unique words change as a function of max number of
# documents a word appears
nr_docs = 10e0**np.linspace(0, 7, num=8)
max_df = (nr_docs+0.5)/len(corpus)
nr_words_all = np.zeros(len(nr_docs))
nr_word_nonr = np.zeros(len(nr_docs))
nr_words_en = np.zeros(len(nr_docs))
nr_words_lemmatized = np.zeros(len(nr_docs))
nr_words_stemmed = np.zeros(len(nr_docs))

for i in range(len(nr_docs)):
    vectorizer = CountVectorizer(max_df=max_df[i])
    vec = vectorizer.fit_transform(corpus)
    nr_words_all[i] = np.shape(vec)[1]
    words = vectorizer.get_feature_names()
    # remove words with special characters and numbers in them
    words_nonr = [word for word in words if word.isalpha()]
    nr_word_nonr[i] = len(words_nonr)
    # correctly spellend english words
    words_en = [word for word in words_nonr if dictionary.check(word)]
    nr_words_en[i] = len(words_en)
    # lemmatize
    lemmas = list(set([lemmatizer.lemmatize(word) for word in words_en]))
    nr_words_lemmatized[i] = len(lemmas)
    # stem
    stems = list(set([stemmer.stem(word) for word in words_en]))
    nr_words_stemmed[i] = len(stems)

    print(nr_docs[i], nr_words_all[i], nr_word_nonr[i],
          nr_words_en[i], nr_words_lemmatized[i], nr_words_stemmed[i])

plt1, = plt.plot(nr_docs, nr_words_all)
plt2, = plt.plot(nr_docs, nr_word_nonr)
plt3, = plt.plot(nr_docs, nr_words_en)
plt4, = plt.plot(nr_docs, nr_words_lemmatized)
plt5, = plt.plot(nr_docs, nr_words_stemmed)
plt.legend([plt1, plt2, plt3, plt4, plt5],
           ['all words', 'words with only chars.',
            'correctly spelled words', 'lemmas', 'stems'],
           loc=4, fontsize=10)

plt.semilogx()
plt.xlabel('max # documents')
plt.ylabel('# unique words')
plt.savefig('../images/nr_words_vs_max_docs.png', dpi=150)
plt.show()
plt.close()
