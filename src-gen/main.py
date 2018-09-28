import raw_corpus2tsv
import preprocess
import mallet
import rank_documents


# change switches to trigger different parts of pipeline
to_tsv = False
custom_prep = False
prepare_data = False
topic_model = True
n_topics = [50]
rank_documents = False
percent = [.1]
min_freq = 0.1

# set the paths
# path = '/gpfs/data/datasci/paper-m/'


# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
# ----------------- WRITE YOUR OWN data2tsv() and prepare_custom() FUNCTIONS
# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
if to_tsv == True:
    # convert raw data to input tsv format
    raw_corpus2tsv.xml2tsv("../test/")

if custom_prep == True:
    # do data prep custom to hansard
    text = preprocess.prepare_custom()
# ----------------- THE STEPS ABOVE ARE SPECIFIC TO EACH DATA SET -----------------

if prepare_data == True:
    # preprocess tsv data for topic modeling
    preprocess.prepare_data(text)        

if topic_model == True:
    # import preprocessed data to mallet objects and train LDA model
    for topic in n_topics:
        mallet.mallet(n_topics)

if rank_documents == True:
    # rank documents by chosen topic(s)
    for p in percent:
        rank_documents.subcorpus(percent, min_freq)