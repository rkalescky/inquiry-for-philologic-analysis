import numpy as np
import pandas as pd
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
import multiprocessing
import enchant
import config

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


def tag2pos(tag, returnNone=False):
    ap_tag = {'NN': wn.NOUN, 'JJ': wn.ADJ,
              'VB': wn.VERB, 'RB': wn.ADV}
    try:
        return ap_tag[tag[:2]]
    except:
        return None if returnNone else ''


def lemmatize_df(df):
    # lemmatize speech acts
    for index, row in df.iterrows():
        # initialize lemma list
        lemma_list = []
        # tokenize word_pos
        if type(row['SPEECH_ACT']) == str:
            tokens = word_tokenize(row['SPEECH_ACT'])
            alpha_tokens = [token for token in tokens if token.isalpha()]
            spellchecked_tokens = [token for token in alpha_tokens
                                   if dictionary.check(token)]
            tagged_tokens = pos_tag(spellchecked_tokens)
            for tagged_token in tagged_tokens:
                word = str(tagged_token[0])
                word_pos = tagged_token[1]
                word_pos_morphed = tag2pos(word_pos)
                if word_pos_morphed is not '':
                    lemma = lemmatizer.lemmatize(word, word_pos_morphed)
                else:
                    lemma = lemmatizer.lemmatize(word)
                lemma_list.append(lemma)
            lemma_string = ' '.join(lemma_list)
            df.loc[index, 'LEMMAS'] = lemma_string
        else:
            print "Not string"
            df.loc[index, 'SPEECH_ACT'] = 'not string'
            df.loc[index, 'LEMMAS'] = 'not string'
            continue

    return(df)
    # df.to_csv("/Users/alee35/land-wars-devel-data/03.lemmatized_speech_acts/membercontributions-lemmatized.tsv", sep="\t")


# load British English spell checker
dictionary = enchant.Dict("en_GB")
# lemmatizer
lemmatizer = WordNetLemmatizer()

# create as many processes as there are CPUs on your machine
num_processes = multiprocessing.cpu_count()
# calculate the chunk size as an integer
chunk_size = int(config.text.shape[0]/num_processes)
# works even if the df length is not evenly divisible by num_processes
chunks = [config.text.ix[config.text.index[i:i + chunk_size]]
          for i in range(0, config.text.shape[0], chunk_size)]
# create our pool with `num_processes` processes
pool = multiprocessing.Pool(processes=num_processes)
# apply our function to each chunk in the list
result = pool.map(lemmatize_df, chunks)
# combine the results from our pool to a dataframe
config.textlem = pd.DataFrame().reindex_like(config.text)
config.textlem['LEMMAS'] = np.NaN
for i in range(len(result)):
    config.textlem.ix[result[i].index] = result[i]


# write speech acts to files for triplet tagging
for index, row in config.textlem.iterrows():
    f = '/Users/alee35/Google Drive/repos/Stanford-OpenIE-Python/hansard/debate_{}.txt'.format(index)
    with open(f, 'w') as f:
        f.write(row['LEMMAS'])


# create year and decade columns
config.textlem['YEAR'] = config.textlem.DATE.str[:4]
config.textlem['DECADE'] = config.textlem['YEAR'].map(lambda x: int(x) - (int(x) % 10))

# groupby debates and concatenate
config.debates = config.textlem.groupby(['YEAR', 'BILL']).aggregate({'LEMMAS': lambda x: x.str.cat(sep='. ')}).reset_index()
for index, row in config.debates.iterrows():
    f = '/Users/alee35/Google Drive/repos/Stanford-OpenIE-Python/hansard/debate_{}.txt'.format(index)
    with open(f, 'w') as f:
        f.write(row['LEMMAS'])
