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
    return sum([value_of(tag) for sentence in dict_tagged_sentences
                for token in sentence for tag in token[2]])


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

# raw speech acts
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
config.textsent = pd.read_csv(config.path_input + 'hansard_sentiment.tsv',
                              sep='\t')

# fill sentiment scores with 0 if speech act wasn't a string
config.textsent['SENTIMENT_SCORE'] = config.textsent['SENTIMENT_SCORE'].fillna(0)
# replace speech acts that aren't string with 'no text'
for index, row in config.textsent.iterrows():
    if type(row['SPEECH_ACT']) != str:
        config.textsent.loc[index, 'SPEECH_ACT'] = 'no text'

# get year from date
config.textsent['YEAR'] = config.textsent.DATE.str[:4]
# convert years column to numeric
config.textsent['YEAR'] = config.textsent['YEAR'].astype(float)
# change years after 1908 to NaN
for index, row in config.textsent.iterrows():
    if row['YEAR'] > 1908:
        config.textsent.loc[index, 'YEAR'] = np.NaN
# forward fill missing dates
config.textsent['YEAR'] = config.textsent['YEAR'].fillna(method='ffill')
# compute decade
config.textsent['DECADE'] = (config.textsent['YEAR'].
                         map(lambda x: int(x) - (int(x) % 10)))

# groupby debates and concatenate, get average/sum speech act sentiments
config.debates = (config.textsent.groupby(['YEAR', 'BILL']).
                  aggregate({'SPEECH_ACT': lambda x: x.str.cat(sep='. '),
                             'SENTIMENT_SCORE': {'AVG_SENTIMENT': 'mean',
                                                 'STD_SENTIMENT': 'std',
                                                 'SUM_SENTIMENT': 'sum'}}).
                  reset_index())

# groupby year and get average sentiment for each year's worth of speech acts
config.yearly_sent = (config.textsent.groupby(['YEAR']).
                      aggregate({'SENTIMENT_SCORE': {'AVG_SENTIMENT': 'mean',
                                                     'STD_SENTIMENT': 'std'}}).
                      reset_index())

# groupby decade and get min/max sentiment speech acts
decade_min = config.textsent.loc[config.textsent.groupby('DECADE')['SENTIMENT_SCORE'].idxmin()]
decade_max = config.textsent.loc[config.textsent.groupby('DECADE')['SENTIMENT_SCORE'].idxmax()]
# get all emotional words in min decades
decade_min['EMOTIONAL_WORDS'] = 's'
for index, row in decade_min.iterrows():
    # set up data structure
    if type(row['SPEECH_ACT']) == str:
        split_sentences = splitter.split(row['SPEECH_ACT'])
    else:
        print "Not string"
        continue
    # pos tag each word in the sentences
    pos_tagged_sentences = postagger.pos_tag(split_sentences)
    # tag sentences with sentiment
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # append emotional words to a list
    emotional_words = []
    for sentence in dict_tagged_sentences:
        for word in sentence:
            if len(word[2]) > 1:
                emotional_words.append(word)
    decade_min.set_value(index, 'EMOTIONAL_WORDS', emotional_words)
# get all emotional words in max decades
decade_max['EMOTIONAL_WORDS'] = 's'
for index, row in decade_max.iterrows():
    # set up data structure
    if type(row['SPEECH_ACT']) == str:
        split_sentences = splitter.split(row['SPEECH_ACT'])
    else:
        print "Not string"
        continue
    # pos tag each word in the sentences
    pos_tagged_sentences = postagger.pos_tag(split_sentences)
    # tag sentences with sentiment
    dict_tagged_sentences = dicttagger.tag(pos_tagged_sentences)
    # append emotional words to a list
    emotional_words = []
    for sentence in dict_tagged_sentences:
        for word in sentence:
            if len(word[2]) > 1:
                emotional_words.append(word)
    decade_max.set_value(index, 'EMOTIONAL_WORDS', emotional_words)
decade_min.to_csv('../data/decade_min.tsv', sep='\t')
decade_max.to_csv('../data/decade_max.tsv', sep='\t')

# plot average sentiment for each debate
plt.scatter(config.debates.YEAR,
            config.debates.SENTIMENT_SCORE.AVG_SENTIMENT,
            alpha=0.1)
plt.title('Debate')
plt.ylabel('sentiment')
plt.xlabel('years')
plt.grid(True)
# plt.savefig('../images/hansard_sentiment_avg_debate.jpg')
plt.show()

# plot sentiment for each speech_act
plt.scatter(config.textsent.YEAR, config.textsent.SENTIMENT_SCORE, alpha=0.1)
plt.title('Speech Act')
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
# add vertical lines for election years
xcoords = [1802, 1806, 1807, 1812, 1818, 1820, 1826, 1830, 1831, 1832, 1835,
           1837, 1841, 1847, 1852, 1857, 1859, 1865, 1868, 1874, 1880, 1885,
           1886, 1892, 1895, 1900, 1906]
for xc in xcoords:
    plt.axvline(x=xc, alpha=0.5)
plt.title('Average Yearly Speech Act')
plt.ylabel('average sentiment')
plt.xlabel('years')
plt.grid(False)
plt.savefig('../images/hansard_sentiment_avg_speechact_error.jpg')
plt.show()

# plot min and max
plt.scatter(decade_min.DECADE, decade_min.SENTIMENT_SCORE, color='r')
plt.scatter(decade_max.DECADE, decade_max.SENTIMENT_SCORE)
# plot average
plt.plot(config.yearly_sent.YEAR,
         config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT)
# plot std
fill_high = (config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT +
             config.yearly_sent.SENTIMENT_SCORE.STD_SENTIMENT)
fill_low = (config.yearly_sent.SENTIMENT_SCORE.AVG_SENTIMENT -
            config.yearly_sent.SENTIMENT_SCORE.STD_SENTIMENT)
plt.fill_between(config.yearly_sent.YEAR, fill_low, fill_high, alpha=0.25)
plt.title('Emotional Extremes')
plt.ylabel('sentiment')
plt.xlabel('decade')
plt.grid(True)
plt.savefig('../images/min_max_sentiment.jpg')
plt.show()
