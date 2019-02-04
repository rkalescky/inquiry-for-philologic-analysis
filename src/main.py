import os
import raw_corpus2tsv
import preprocess
import mallet


# change switches to trigger different parts of pipeline
to_tsv = False
custom_prep = True
data_dt = '20181210'
pipeline_prepare_data = True
mallet_import = False
topic_model = False
n_topics = [50]
rank_documents = False
n_docs = [100]
topic_idx = [1,10,20,30]



# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
# ----------------- WRITE YOUR OWN data2tsv() and prepare_custom() FUNCTIONS
# ----------------- THE STEPS BELOW ARE SPECIFIC TO EACH DATA SET -----------------
if to_tsv == True:
    # convert raw data to input tsv format
    raw_corpus2tsv.xml2tsv("../test/")

if custom_prep == True:
    # do data prep custom to hansard
    text = preprocess.prepare_custom(data_dt)
# ----------------- THE STEPS ABOVE ARE SPECIFIC TO EACH DATA SET -----------------

if pipeline_prepare_data == True:
    # preprocess tsv data for topic modeling
    preprocess.pipeline_prepare_data(text)        

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
