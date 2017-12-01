import config


def agg_DC(dataframe):
    '''
    calculate fraction derived corpus per year
    '''
    years = dataframe.groupby(0).size().reset_index(name='count')
    yearsagg = years_all.merge(years, how='outer', on=0).fillna(value=0)
    yearsagg['diff'] = yearsagg['count_x']-yearsagg['count_y']
    yearsagg['fraction'] = yearsagg['count_y']/yearsagg['count_x']
    yearsagg.columns = ['year', 'num_debates', 'num_derived',
                           'diff', 'fraction']
    return(years, yearsagg)


years_all = config.metadata.groupby(0).size().reset_index(name='count')
years1, yearsagg1 = agg_DC(config.concat_df1)
years5, yearsagg5 = agg_DC(config.concat_df5)
years10, yearsagg10 = agg_DC(config.concat_df10)
years15, yearsagg15 = agg_DC(config.concat_df15)
years20, yearsagg20 = agg_DC(config.concat_df20)
years25, yearsagg25 = agg_DC(config.concat_df25)
years_overlap, yearsagg_overlap = agg_DC(config.concat_overlap)

# fraction derived corpus per year
plt.bar(yearsagg_overlap['year'], yearsagg_overlap['fraction'], align='center',
        width=1, color='b', alpha=1, label='overlap')
plt.bar(yearsagg1['year'], yearsagg1['fraction'], align='center',
        width=1, color='b', alpha=.4, label='1% most similar')
plt.bar(yearsagg5['year'], yearsagg5['fraction'], align='center',
        width=1, color='b', alpha=.3, label='5% most similar')
plt.bar(yearsagg10['year'], yearsagg10['fraction'], align='center',
        width=1, color='b', alpha=.2, label='10% most similar')
plt.bar(yearsagg15['year'], yearsagg15['fraction'], align='center',
        width=1, color='b', alpha=.1, label='15% most similar')
plt.bar(yearsagg20['year'], yearsagg20['fraction'], align='center',
        width=1, color='b', alpha=.05, label='20% most similar')
plt.bar(yearsagg25['year'], yearsagg25['fraction'], align='center',
        width=1, color='b', alpha=.05, label='25% most similar')
plt.title('Fraction Derived Corpus (KLD1)')
plt.ylabel('fraction debates')
plt.xlabel('years')
plt.legend(bbox_to_anchor=(1, 0),
           loc='lower right', ncol=1, fontsize='xx-small')
plt.grid(True)
plt.savefig('../images/fr_overlapping_debates_year_KLD1.jpg')
plt.show()

# difference derived corpus per year
plt.bar(yearsagg_overlap['year'], yearsagg_overlap['diff'],
        align='center', width=1, color='b', alpha=.25, label='overlap')
plt.bar(yearsagg1['year'], yearsagg1['diff'],
        align='center', width=1, color='b', alpha=.3, label='1% most similar')
plt.bar(yearsagg5['year'], yearsagg5['diff'],
        align='center', width=1, color='b', alpha=.4, label='5% most similar')
plt.bar(yearsagg10['year'], yearsagg10['diff'],
        align='center', width=1, color='b', alpha=.5, label='10% most similar')
plt.bar(yearsagg15['year'], yearsagg15['diff'],
        align='center', width=1, color='b', alpha=.6, label='15% most similar')
plt.bar(yearsagg20['year'], yearsagg20['diff'],
        align='center', width=1, color='b', alpha=1, label='20% most similar')
plt.bar(yearsagg25['year'], yearsagg25['diff'], align='center',
        width=1, color='b', alpha=1, label='25% most similar')
plt.title('Difference Derived Corpus (KLD1)')
plt.ylabel('number debates')
plt.xlabel('years')
plt.legend(bbox_to_anchor=(1, 1),
           loc='upper right', ncol=1, fontsize='xx-small')
plt.grid(True)
plt.savefig('../images/diff_overlapping_derived_year_KLD1.jpg')
plt.show()

# number derived corpus per year
plt.bar(years_overlap[0], years_overlap['count'], align='center', width=1,
        color='b', alpha=1, label='overlap')
plt.bar(years1[0], years1['count'], align='center', width=1,
        color='b', alpha=.4, label='1% most similar')
plt.bar(years5[0], years5['count'], align='center', width=1,
        color='b', alpha=.3, label='5% most similar')
plt.bar(years10[0], years10['count'], align='center', width=1,
        color='b', alpha=.2, label='10% most similar')
plt.bar(years15[0], years15['count'], align='center', width=1,
        color='b', alpha=.1, label='15% most similar')
plt.bar(years20[0], years20['count'], align='center', width=1,
        color='b', alpha=.05, label='20% most similar')
plt.bar(years25[0], years25['count'], align='center', width=1,
        color='b', alpha=0.01, label='25% most similar')
plt.title("Number Derived Corpus (KLD1)")
plt.ylabel("number debates")
plt.xlabel("years")
plt.legend(bbox_to_anchor=(0, 1),
           loc='upper left', ncol=1, fontsize='xx-small')
plt.grid(True)
plt.savefig("../images/nr_overlap_derived_year_KLD1.jpg")
plt.show()

# plot nr overlapping debates between time periods
p = config.overlap_nums
plt.figure(figsize=(10, 3))
plt.bar(p.index, p['num_overlap'], align='center', width=1,
        color='b', alpha=1, label='overlapping debates')
plt.bar(p.index, p['num_total'], align='center', width=1,
        color='b', alpha=0.25, label='all debates')
xticks = p['years']
plt.xticks(p.index, xticks, rotation=90)
plt.title("Overlap between Adjacent 5-year Time Windows")
plt.ylabel("number debates")
plt.xlabel("time windows")
plt.legend(bbox_to_anchor=(0, 1),
           loc='upper left', ncol=1, fontsize='xx-small')
plt.grid(True)
plt.savefig("../images/nr_overlap_timewindows_KLD1.jpg")
plt.show()

# get top 5 years - get top debates in each year
top_years = (yearsagg_overlap[yearsagg_overlap['fraction'] > 0.95]['year'].
             values.tolist())
tydebates = config.concat_overlap[config.concat_overlap[0].isin(top_years)]
config.top
