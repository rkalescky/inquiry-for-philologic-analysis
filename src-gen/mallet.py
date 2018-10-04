import os

# load the mallet module
os.system("module load mallet/2.0.8rc3")


def imprt():
    # import the data into mallet format
    os.system("sh mallet_import_from_file.sh") 

def lda(n):
    # train the lda model with x number of topics
    os.system("sh mallet_train_lda.sh " + str(n))