import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import matplotlib
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import enchant

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# load pyenchant english dictionary
dictionary = enchant.Dict("en_US")
# load nltk lemmatizer
lemmatizer = WordNetLemmatizer()

# read the excel file
df = pd.read_excel('data/data_science_extract_nopw.xlsx')
# replace nans with empty string
df.replace(np.nan, '', regex=True,inplace=True)
# merge essays and answers and prepare the corpus
corpus = list(df['Essay question'] + df['Area of Study'] + df['Why  Brown?'] + df['Where you have lived?'] + \
              df['Communities or Groups'])

# explore how the number of unique words change as a function of max number of documents a word appears
nr_docs = 10e0**np.linspace(0,5,num=6)
max_df = (nr_docs+0.5)/len(corpus)
nr_words_all = np.zeros(len(nr_docs))
nr_word_nonr = np.zeros(len(nr_docs))
nr_words_en = np.zeros(len(nr_docs))
nr_words_lemmatized = np.zeros(len(nr_docs))

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

    # lemmatize
    #vectorizer = CountVectorizer(max_df=max_df[i],tokenizer=LemmaTokenizer())
    #vec = vectorizer.fit_transform(corpus)

    print nr_docs[i],nr_words_all[i],nr_word_nonr[i],nr_words_en[i],nr_words_lemmatized[i]

plt1, = plt.plot(nr_docs,nr_words_all)
plt2, = plt.plot(nr_docs,nr_word_nonr)
plt3, = plt.plot(nr_docs,nr_words_en)
plt4, = plt.plot(nr_docs,nr_words_lemmatized)
plt.legend([plt1,plt2,plt3,plt4],['all words','words with only chars.','correctly spelled words',\
                                  'lemmas'],loc=2,fontsize=11)

plt.semilogx()
plt.xlabel('max #documents')
plt.ylabel('#unique words')
plt.savefig('figures/nr_words_vs_max_docs.png',dpi=150)
plt.show()
plt.close()
