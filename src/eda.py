import pandas as pd
import matplotlib as plt


def fraction_derived_corpus(dataframe):
    '''
    calculate fraction derived corpus per year
    '''
    years = dataframe.groupby(0).size().reset_index(name="count")
    years_percs = years_all.merge(years, how='outer', on=0).fillna(value=0)
    years_percs['fraction'] = years_percs['count_y']/years_percs['count_x']
    years_percs.columns = ["year", "num_debates", "num_derived", "fraction"]
    return(years, years_percs)


# read metadata file
with open('/Users/alee35/land-wars-devel-data/04stemmed_bills/long_bills_stemmed_metadata.tsv', 'r') as f:
    metadata = pd.read_csv(f, sep='\t', header=None)

# number debates per year
years_all = metadata.groupby(0).size().reset_index(name = "count")

# Per Year
plt.bar(years_all[0], years_all['count'], align='center', width=1)
plt.title("19th Century Hansards")
plt.ylabel("nr. debates")
plt.xlabel("years")
plt.grid(True)
plt.savefig("./output/DC_output/nr_debates_year.jpg")
plt.show()

# fraction debates per year
years_1850, years_percs_1850 = fraction_derived_corpus(dc2_1850)


plt.bar(years_percs_1850["year"], years_percs_1850['fraction'], align = 'center', width = 1, color='b', alpha=1, label='1% most similar')
plt.title("Fraction Derived Corpus (KLD1, time window = 1850-1854)")
plt.ylabel("fr. debates")
plt.xlabel("years")
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1, fontsize = 'xx-small')
plt.grid(True)
plt.savefig("./output/DC_output/kld1/fr_derived_year_kld1.jpg")
plt.show()
