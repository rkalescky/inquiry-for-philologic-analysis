import matplotlib.pyplot as plt
import config
from wordcloud import WordCloud


# test wordcloud on 1% threshold
# lowercase debate titles and concatenate to giant text string
text = config.concat_df1[2].apply(lambda x: x.lower())
text = text.str.cat(sep=' ')

# create the wordcloud
wordcloud = WordCloud().generate(text)

# generate the image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('../images/wordcloud_{}.jpg'.format(1))
plt.show()


# save wordclouds for each threshold
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
