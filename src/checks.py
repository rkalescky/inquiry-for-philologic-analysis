import config

path = '~/land-wars-devel-data/04stemmed_bills/'
seed_comp = pd.read_csv(path + 'seed_composition_500.txt',
                        sep='\t', header=None)
doc_comp = pd.read_csv(path + 'mallet_composition_500.txt',
                       sep='\t', header=None, usecols=range(2,502,1))
doc_keys = pd.read_csv(path + 'keys_500_topics_sorted.txt',
                       sep='\t')
land_topics = [1,406,362,415,152,65,235,389,244,5,129,153,262,135,84,453,216,247,156]

# check the probabilities of Jo's hand-selected land topics in the seed
seed_check = seed_comp.iloc[:, land_topics]
seed_check.to_csv(path + 'land_topics_check.csv', index=False)

# histogram the probabilities for each seed
# - Napier, Devon, Richmond, Bessborough
plt.hist(seed_comp.iloc[0,2:])
plt.ylabel('Frequency')
plt.xlabel('Topic Probability')
plt.title('Napier Report')
plt.show()

plt.hist(seed_comp.iloc[1,2:])
plt.ylabel('Frequency')
plt.xlabel('Topic Probability')
plt.title('Devon Report')
plt.show()

plt.hist(seed_comp.iloc[2,2:])
plt.ylabel('Frequency')
plt.xlabel('Topic Probability')
plt.title('Richmond Report')
plt.show()

plt.hist(seed_comp.iloc[3,2:])
plt.ylabel('Frequency')
plt.xlabel('Topic Probability')
plt.title('Bessborough Report')
plt.show()

# what are the topics for the top 10 most prevalent topics in the seed
seed_array = seed_comp.iloc[:, 2:].as_matrix()
seed_ind_n = np.argpartition(seed_array[0], -20)[-20:]
seed_array[0][seed_ind_n]

# does the Dirichlet parameter sum to 1? No.
doc_keys['Relative Frequency'].sum()

# do the document probabilities sum to 1?
for index, row in doc_comp.iterrows():
    print row.sum()

# what are the top 1% documents for each topic
threshold = math.floor(len(doc_check) * 0.01)
doc_check = doc_comp.iloc[:, land_topics]
doc_matrix = doc_check.iloc[:,0].as_matrix()
idx = np.argpartition(doc_matrix, )
