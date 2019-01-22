import time
import numpy as np
import pandas as pd
import pickle
import scipy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


# # Figure 3
# plt.figure(figsize=(10,7))

# dt = '20181212'
# df = pd.read_csv('~/Dropbox (Brown)/data/hansard/dhq/mc-stemmed{}.txt'.format(dt), delimiter='\t', usecols=[2], header=None)
# print(list(df))

# # replace nans with empty string
# df.replace(np.nan, '', regex=True, inplace=True)
# # prepare the corpus
# corpus = list(df[2])

# # prepare the word counts
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)
# counts = X.getnnz(axis=0)

# plt.hist(counts, bins = 10 ** np.linspace(np.log10(1), np.log10(1000000), 30))
# plt.xscale('log')
# plt.yscale('log')
# plt.xlabel('# document')
# plt.ylabel('# unigrams')
# plt.title('Hansard Unigram Frequencies')
# plt.grid(True, alpha=0.2)
# plt.savefig('./images/hist_words_docs_tight.png', dpi=300, bbox_inches='tight')
# plt.show()
# plt.close()



# Figure 2
fig, axes = plt.subplots(figsize=(10,7), nrows=2, ncols=2, sharex=True)

# number debates per year
with open("/users/alee35/repos/land-wars/ashley/long_bills_stemmed_metadata.tsv", 'r') as f:
    metadata = pd.read_csv(f, sep = '\t', header = None)
years_all = metadata.groupby(0).size().reset_index(name = "count")
axes[0,0].bar(years_all[0], years_all['count'], align = 'center', width = 1)
axes[0,0].set_title("Debates Per Year")
axes[0,0].set_ylabel("# debates")
axes[0,0].grid(True, alpha=0.2)

dt = '20181210'
mc = pd.read_csv('~/Dropbox (Brown)/data/hansard/dhq/membercontributions-{}.tsv'.format(dt), delimiter='\t')
mc['YEAR'] = mc['DATE'].str[:4]
mc['YEAR'] = mc['YEAR'].astype('float')
for index, row in mc.iterrows():
    # fix years after 1908
    if row['YEAR'] > 1908:
        mc.loc[index, 'YEAR'] = np.NaN
# forward fill missing dates
mc['YEAR'].fillna(method='ffill', inplace=True)
speakers = mc.groupby(['YEAR'])['MEMBER'].nunique().reset_index(name='count')
axes[0,1].bar(speakers['YEAR'], speakers['count'], align = 'center', width=1)
axes[0,1].set_title("Members of Parliament")
axes[0,1].set_ylabel("# speakers")
axes[0,1].grid(True, alpha=0.2)

# average number of speech acts per debate per year (standard error bars)
avg_sa = mc.groupby(['BILL', 'YEAR']).size().reset_index(name='count')
avg_sa_yr = avg_sa.groupby(['YEAR'], as_index=False).agg({'count':['mean', 'std', percentile(16), percentile(84)]})
avg_sa_yr.columns = ['year', 'mean', 'std', 'percentile_16', 'percentile_84']
axes[1,0].errorbar(avg_sa_yr['year'], avg_sa_yr['mean'], yerr=[avg_sa_yr['percentile_16'], avg_sa_yr['percentile_84']], markerfacecolor='black', markeredgecolor='black', markersize=1, fmt='o', elinewidth=1, ecolor='gray')
axes[1,0].set_title("Average Speech Acts Per Debate")
axes[1,0].set_ylabel("# speech acts")
axes[1,0].set_xlabel("years")
axes[1,0].grid(True, alpha=0.2)

# average number of words per debate per year (standard error bars)
dt = '20181212'
stem = pd.read_csv('~/Dropbox (Brown)/data/hansard/dhq/mc-stemmed{}.txt'.format(dt), delimiter='\t', header=None, usecols=[1,2])
stem['word_count'] = stem[2].apply(lambda x: len(str(x).split(' ')))
avg_words_yr = stem.groupby([1], as_index=False).agg({'word_count':['mean', 'std', percentile(16), percentile(84)]})
avg_words_yr.columns = ['year', 'mean', 'std', 'percentile_16', 'percentile_84']
avg_words_yr = avg_words_yr[pd.to_numeric(avg_words_yr['year'], errors='coerce').notnull()]
avg_words_yr['year'] = avg_words_yr['year'].astype('float')
avg_words_yr = avg_words_yr.ix[avg_words_yr['mean']<40000,:]
axes[1,1].errorbar(avg_words_yr['year'], avg_words_yr['mean'], yerr=[avg_words_yr['percentile_16'], avg_words_yr['percentile_84']], markerfacecolor='black', markeredgecolor='black', markersize=1, fmt='o', elinewidth=1, ecolor='gray')
axes[1,1].set_title("Average Unigrams Per Debate")
axes[1,1].set_ylabel("# unigrams")
axes[1,1].set_xlabel("years")
axes[1,1].grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('./images/hansard_overview_tight.png', dpi=300)
plt.show()
plt.close()



# ---------------------------------- 
# find speakers for top documents in z=500 rankings

date_mc = '20181210'
date_midprep = '20181211'

# read in slightly cleaned up speech acts
# with open('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/mc-midprep-' + date_midprep + '.tsv', 'r') as f:
#     text = pd.read_csv(f, sep='\t')

with open('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/membercontributions-' + date_mc + '.tsv', 'r') as f:
    text = pd.read_csv(f, sep='\t')
    
print('document read in successfully!')
print(text.isnull().sum())
print(list(text))


def get_metadata(title, date, members):
    a = text.ix[(text['BILL'].str.contains(title)) & (text['DATE'] == date), ['BILL', 'MEMBER', 'DATE', 'SPEECH_ACT']]
    if a.shape[0] > 0:
        print(a.shape)
        # a.to_csv('/Users/alee35/Dropbox (Brown)/data/hansard/dhq/{}_{}.tsv'.format(title, date), sep='\t', index=False)
        memb = list(a.MEMBER.unique())
        for m in memb:
            members.append(m)
    else:
        b = text.ix[(text['DATE'] == date), ['BILL', 'MEMBER', 'DATE', 'SPEECH_ACT']]
        print('------------------------')
        bills = list(b.BILL.unique())
        for bill in bills:
            print(bill)
        print('------------------------')


# z = 500
members_500 = []
get_metadata('COMMUTATION OF TITHES', '1836-03-25', members_500)
get_metadata('LEASEHOLD S ENFRANCHISEMENT', '1889-05-01', members_500)
get_metadata('SECOND READING', '1890-03-27', members_500)
get_metadata('TITHE RENT-CHARGE RECOVERY', '1889-08-12', members_500)
get_metadata('TITHE RENT-CHARGE RECOVERY', '1889-08-13', members_500)
get_metadata('COMMUTATION OF TITHES, \(ENGLAND.\)', '1835-03-24', members_500)
get_metadata('TITHES \(IRELAND\) MINISTBEIAL PLAN', '1832-07-05', members_500)
get_metadata('TENANTS IN TOWNS IMPROVEMENT \(IRELAND\) BILL', '1900-04-04', members_500)
get_metadata('TITHES \(IRELAND\)', '1834-05-02', members_500)
get_metadata('TITHES \(IRELAND\)', '1834-02-20', members_500)
# get_metadata('COMMITTEE', '1890-06-05')
# get_metadata('SECOND READING ADJOURNED DEBATE', '1890-03-28')
# get_metadata('COMMITTEE', '1887-07-25')

# z = 0
members_0 = []
get_metadata('IRISH LAND COMMISSION', '1897-06-29', members_0)
get_metadata('IRISH LAND COMMISSION', '1897-06-25', members_0)
get_metadata('IRISH LAND COMMISSION FAIR RENTS, CO. WESTMEATH.', '1888-04-30', members_0)
get_metadata('FAIR RENT APPEALS IN COUNTY ANTRIM', '1899-07-27', members_0)
get_metadata('LAND COMMISSION \(KING\'S COUNTY\)', '1897-02-11', members_0)
get_metadata('Fair Rent Cases in County Roscommon', '1904-03-22', members_0)
get_metadata('COUNTY DOWN LAND COMMISSION', '1900-02-16', members_0)
get_metadata('JUDICIAL RENTS \(COUNTY MONAGHAN\)', '1896-08-11', members_0)
get_metadata('North Tipperary Land Court', '1904-03-07', members_0)
get_metadata('Listowel Fair Rent Applications', '1908-07-31', members_0)
get_metadata('FERMANAGH RENT APPEALS', '1901-03-04', members_0)
get_metadata('IRISH LAND COMMISSION WEXFORD', '1888-03-05', members_0)
get_metadata('FAIR RENT APPEALS IN CORK', '1900-07-24', members_0)
get_metadata('CORK LAND COMMISSION', '1900-07-27', members_0)
# get_metadata('NEXT SITTING  AT LONGFORD OF APPEAL COURT OF LAND COMMISSION', '1906-11-14')
# get_metadata('MIDLETON FAIR RENT APPLICATIONS', '1907-02-14')

print('-----# of members in z500: {}-----'.format(len(members_500)))
for member in members_500:
    print(member)
print('-----# of members in z500: {}-----'.format(len(members_500)))


print('-----# of members in z0: {}-----'.format(len(members_0)))
for member in members_0:
    print(member)

print('-----# of members in z0: {}-----'.format(len(members_0)))