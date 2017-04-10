import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import config

# code from http://fjavieralba.com/basic-sentiment-analysis-with-python.html


class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        '''
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'],
                   ['this', 'is', 'another', 'one']]
        '''
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent)
                               for sent in sentences]
        return tokenized_sentences


class POSTagger(object):
    def __init__(self):
        pass

    def pos_tag(self, sentences):
        '''
        input format: list of lists of words
            e.g.: [['this', 'is', 'a', 'sentence'],
                   ['this', 'is', 'another', 'one']]
        output format: list of lists of tagged tokens. Each tagged tokens has a
        form, a lemma, and a list of tags
            e.g: [[('this', 'this', ['DT']), ('is', 'be', ['VB']),
                   ('a', 'a', ['DT']), ('sentence', 'sentence', ['NN'])],
                    [('this', 'this', ['DT']), ('is', 'be', ['VB']),
                     ('another', 'another', ['DT']), ('one', 'one', ['CARD'])]]
        '''
        pos = [nltk.pos_tag(sentence) for sentence in sentences]
        # adapt format
        pos = [[(word, word, [postag]) for (word, postag) in sentence]
               for sentence in pos]
        return pos


class DictionaryTagger(object):
    def __init__(self, dictionary_list):
        dictionaries = [df for df in dictionary_list]
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))
        print dictionaries

    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence)
                for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        '''
        the result is only one tagging of all the possible ones.
        The resulting tagging is determined by these two priority rules:
            - longest matches have higher priority
            - search is made from left to right
        '''
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N)    # avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for
                                            word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for
                                             word in sentence[i:j]]).lower()
                if tag_with_lemmas:
                    literal = expression_lemma
                else:
                    literal = expression_form
                if literal in self.dictionary:
                    # self.logger.debug('found: %s' % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form,
                                         expression_lemma, taggings)
                    # if the tagged literal is a single token,
                    # conserve its previous taggings:
                    if is_single_token:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence


def value_of(sentiment):
    if sentiment == 'positive':
        return 1
    if sentiment == 'negative':
        return -1
    return 0


def sentiment_score(review):
    return sum([value_of(tag) for sentence in dict_tagged_sentences for token in sentence for tag in token[2]])


splitter = Splitter()
postagger = POSTagger()

# read and process negative and positive adjective lists from Roget's
neg_path = '../data/negative.txt'
pos_path = '../data/positive.txt'
with open(neg_path, 'r') as f:
    neg = pd.read_csv(f, header=None, names=['word'])
with open(pos_path, 'r') as f:
    pos = pd.read_csv(f, header=None, names=['word'])
neg['valence'] = 'negative'
pos['valence'] = 'positive'
# convert valences into lists
neg.valence = neg.valence.map(lambda x: [x])
pos.valence = pos.valence.map(lambda x: [x])
# convert neg and pos dataframes to dictionaries
neg_dict = dict(zip(neg.word, neg.valence))
pos_dict = dict(zip(pos.word, pos.valence))

# set up dictionary of sentiment valences
dicttagger = DictionaryTagger([neg_dict, pos_dict])

# try test raw speech acts
for index, row in config.text2.iterrows():
    # set up data structure
    split_sentences = splitter.split(row['SPEECH_ACT'])
    pos_tagged_sentences = postagger.pos_tag(split_sentences)
    # tag sentences with sentiment
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # compute sentiment score
    sentiment = sentiment_score(dict_tagged_sentences)
    # print sentiment
    config.text2.loc[index, 'SENTIMENT_SCORE'] = sentiment

# try test lemmas
for index, row in config.textlem2.iterrows():
    # set up data structure
    if type(row['LEMMAS']) == str:
        split_sentences = splitter.split(row['LEMMAS'])
    else:
        print "Not string"
        continue
    # pos tag each word in the sentences
    pos_tagged_sentences = postagger.pos_tag(split_sentences)
    # tag sentences with sentiment
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # compute sentiment score
    sentiment = sentiment_score(dict_tagged_sentences)
    # print sentiment
    config.textlem2.loc[index, 'SENTIMENT_SCORE'] = sentiment

# try raw speech acts
for index, row in config.text.iterrows():
    # set up data structure
    if type(row['SPEECH_ACT']) == str:
        split_sentences = splitter.split(row['SPEECH_ACT'])
    else:
        print "Not string"
        continue
    # pos tag words in sentences
    pos_tagged_sentences = postagger.pos_tag(split_sentences)
    # tag sentences with sentiment
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # compute sentiment score
    sentiment = sentiment_score(dict_tagged_sentences)
    # print sentiment
    config.text.loc[index, 'SENTIMENT_SCORE'] = sentiment

# write to csv / read csv if kernel gets interrupted
config.text.to_csv(config.path_input + 'hansard_sentiment.tsv', sep='\t')
config.text = pd.read_csv(config.path_input + 'hansard_sentiment.tsv', sep='\t')

# fill sentiment scores wtih 0 if speech act wasn't a string
config.text['SENTIMENT_SCORE'] = config.text['SENTIMENT_SCORE'].fillna(0)
# replace speech acts that aren't string with 'no text'
for index, row in config.text.iterrows():
    if type(row['SPEECH_ACT']) != str:
        config.text.loc[index, 'SPEECH_ACT'] = 'no text'

# get year from date
config.text['YEAR'] = config.text.DATE.str[:4]
# convert years column to numeric
config.text['YEAR'] = config.text['YEAR'].astype(float)
# change years after 1908 to NaN
for index, row in config.text.iterrows():
    if row['YEAR'] > 1908:
        config.text.loc[index, 'YEAR'] = np.NaN
# forward fill missing dates
config.text['YEAR'] = config.text['YEAR'].fillna(method='ffill')
# compute decade
config.text['DECADE'] = (config.text['YEAR'].
                         map(lambda x: int(x) - (int(x) % 10)))

# groupby debates and concatenate
config.debates = (config.text.groupby(['YEAR', 'BILL']).
                  aggregate({'SPEECH_ACT': lambda x: x.str.cat(sep='. '),
                             'SENTIMENT_SCORE': {'AVG_SENTIMENT': 'mean',
                                                 'STD_SENTIMENT': 'std',
                                                 'SUM_SENTIMENT': 'sum'}}).
                  reset_index())

# groupby year and average
config.yearly_sent = (config.text.groupby(['YEAR']).
                      aggregate({'SENTIMENT_SCORE': {'AVG_SENTIMENT': 'mean',
                                                     'STD_SENTIMENT': 'std'}}).
                      reset_index())

# plot average sentiment for each debate
plt.scatter(config.debates.YEAR,
            config.debates.SENTIMENT_SCORE.AVG_SENTIMENT,
            alpha=0.1)
plt.title('Average Debate Sentiment')
plt.ylabel('average sentiment')
plt.xlabel('years')
plt.grid(True)
# plt.savefig('../images/hansard_sentiment_avg_debate.jpg')
plt.show()

# plot sum sentiment for each debate
plt.scatter(config.debates.YEAR,
            config.debates.SENTIMENT_SCORE.SUM_SENTIMENT,
            alpha=0.1)
plt.title('Debate Sentiment')
plt.ylabel('sentiment')
plt.xlabel('years')
plt.grid(True)
# plt.savefig('../images/hansard_sentiment_debate.jpg')
plt.show()

# plot average sentiment for each speech_act
plt.scatter(config.text.YEAR, config.text.SENTIMENT_SCORE, alpha=0.1)
plt.title('Speech Act Sentiment')
plt.ylabel('sentiment')
plt.xlabel('years')
plt.grid(True)
plt.savefig('../images/hansard_sentiment_speechact.jpg')
plt.show()

# plot average sentiment for each speech act
plt.plot(config.yearly_sent.YEAR,
         config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT)
# add standard deviation
fill_high = (config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT +
             config.yearly_sent.SENTIMENT_SCORE.STD_SENTIMENT)
fill_low = (config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT -
            config.yearly_sent.SENTIMENT_SCORE.STD_SENTIMENT)
plt.fill_between(config.yearly_sent.YEAR, fill_low, fill_high, alpha=0.25)
plt.title('Average Speech Act Sentiment')
plt.ylabel('average sentiment')
plt.xlabel('years')
plt.grid(True)
plt.savefig('../images/hansard_sentiment_avg_speechact_error.jpg')
plt.show()

# # plot average sentiment for each debate
# config.debates.plot.hexbin('YEAR', 'SUM_SENTIMENT')
# plt.title('Average Debate Sentiment')
# plt.ylabel('average sentiment')
# plt.xlabel('years')
# plt.grid(True)
# # plt.savefig('../images/fr_overlapping_debates_year_KLD1.jpg')
# plt.show()

# df = pd.DataFrame(np.random.randn(1000, 2), columns=['a', 'b'])
# df['b'] = df['b'] = df['b'] + np.arange(1000)
# df['z'] = np.random.uniform(0, 3, 1000)
# df.plot.hexbin(x='a', y='b', C='z', reduce_C_function=np.max, gridsize=25)
# plt.show()
