import os
import raw_corpus2tsv
import preprocess
import mallet
import rank_documents


os.system('module load anaconda/3-5.2.0')

# change switches to trigger different parts of pipeline
to_tsv = False
custom_prep = True
data_dt = '20181210'
prepare_data = True
mallet_import = False
topic_model = False
n_topics = [50]
rank_documents = False
n_docs = [100]
topic_idx = [1,10,20,30]

# set the paths
# path = '/gpfs/data/datasci/paper-m/'


# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
# ----------------- WRITE YOUR OWN data2tsv() and prepare_custom() FUNCTIONS
# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
if to_tsv == True:
    # convert raw data to input tsv format
    # raw_corpus2tsv.xml2tsv("../test/")
    raw_corpus2tsv.xml2tsv("/gpfs/data/datasci/paper-m/raw/hansard_xml/")

if custom_prep == True:
    # do data prep custom to hansard
    text = preprocess.prepare_custom(data_dt)
# ----------------- THE STEPS ABOVE ARE SPECIFIC TO EACH DATA SET -----------------

if prepare_data == True:
    # preprocess tsv data for topic modeling
    preprocess.prepare_data(text)        

if mallet_import == True:
    # load the mallet module
    os.system("module load mallet/2.0.8rc3")
    mallet.imprt()

if topic_model == True:
    # load the mallet module
    os.system("module load mallet/2.0.8rc3")
    # import preprocessed data to mallet objects and train LDA model
    for topic in n_topics:
        mallet.lda(n_topics)

if rank_documents == True:
    # rank documents by chosen topic(s)
    for n in n_docs:
        rank_documents.subcorpus(topic_idx, n_docs)