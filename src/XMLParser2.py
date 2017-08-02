from lxml import etree
import pandas as pd
import os
import re
import sys

reload(sys)
sys.setdefaultencoding('utf-8')

files = os.listdir('/gpfs/data/datasci/paper-m/raw/hansard_xml/')
search_words = ['House divided', 'Committee divided', 'on Division', 'Division List',
                'The Ayes and the Noes',]

f = open('/users/ktaylor6/inquiry-for-philologic-analysis/data/HansardVotes.tsv', 'w+')
for xml in files:
    doc = etree.parse('/gpfs/data/datasci/paper-m/raw/hansard_xml/' +xml)
    for words in search_words:
        result = doc.xpath('//*[@id and contains(., $word)]', word = words)
        for elem in result:
            for child in elem.itertext():
                if child is not None and len(re.findall(r'\d+', child))>1 and words in child:
                    votes = re.findall(r'\d+', child)
                    string = str(elem.attrib)[8:-2] + "\t"
                    string += (str(votes[0]) + "\t" + str(votes[1]) + "\t")
                    string += words + "\t"
                    section = elem
                    date = elem
                    while section.getparent() is not None and "section" not in str(section.tag):
                        section = section.getparent()
                    for children in section.iterchildren():
                        if "title" in str(children.tag):
                            if children.text.strip() == "":
                                string += "NA" + "\t"
                            else:
                                line = children.text.strip()
                                for char in '[.]':
                                    line = line.replace(char, '')
                                string += line + "\t"
                    while date.getparent() is not None and "house" not in str(date.tag):
                        date = date.getparent()
                    for children in date.iterchildren():
                        if "format" in str(children.attrib):
                            string += str(children.attrib)[12:-2] + "\n"
                    f.write(string)       
