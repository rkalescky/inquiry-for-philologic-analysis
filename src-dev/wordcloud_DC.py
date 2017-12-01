from wordcloud import WordCloud, STOPWORDS
import config


# test wordcloud on 1% threshold
# lowercase debate titles and concatenate to giant text string
text = config.concat_df1[2].apply(lambda x: x.lower())
text = text.str.cat(sep=' ')

STOPWORDS.add('question')
STOPWORDS.add('bill')
STOPWORDS.add('second')
STOPWORDS.add('reading')

# create the wordcloud
wordcloud = WordCloud().generate(text)

# generate the image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('../images/wordcloud_{}.jpg'.format(1))
plt.show()


# save wordclouds for each threshold 5-25%
concat_tsvs = [config.concat_df5, config.concat_df10,
               config.concat_df15, config.concat_df20, config.concat_df25]

percent = 5
for tsv in concat_tsvs:

    # lowercase debate titles and concatenate to giant text string
    text = tsv[2].apply(lambda x: x.lower())
    text = text.str.cat(sep=' ')

    # create the wordcloud
    wordcloud = WordCloud().generate(text)

    # generate the image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('../images/wordcloud_{}.jpg'.format(percent))
    # plt.show()

    percent += 5


def wordcloud_timewindow(path):
    '''
    save wordclouds for each timewindow
    '''
    for f in glob2.glob(path):
        # load array
        tsv = pd.read_csv(f, sep='\t', header=None)
        # get metric and year_start from filename
        f = os.path.splitext(f)[0]
        f = os.path.basename(f).split('_')
        metric = f[1]
        year = f[2]

        # lowercase debate titles and concatenate to giant text string
        text = tsv[2].apply(lambda x: x.lower())
        text = text.str.cat(sep=' ')

        # create the wordcloud
        wordcloud = WordCloud().generate(text)

        # generate the image
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('../images/wordcloud_0.01_{}_{}.jpg'.format(metric, year))
        # plt.show()


wordcloud_timewindow(config.path_output + 'DC/debates_kld1_*_0.01.txt')
wordcloud_timewindow(config.path_output + 'DC/overlap_*.txt')


def wordcloud_decade(concat_tsv):
    # lowercase debate titles and concatenate to giant text string
    text = concat_tsv
    text[2] = text[2].map(lambda x: x.lower())
    text = (text.groupby([1]).aggregate({2: lambda x: x.str.cat(sep=' ')}).
            reset_index())

    for index, row in text.iterrows():
        # create the wordcloud
        wordcloud = WordCloud().generate(row[2])

        # generate the image
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('../images/wordcloud_{}.png'.format(row[1]))
        # plt.show()


wordcloud_decade(config.concat_overlap)
