from pycorenlp import StanfordCoreNLP
import json
import config

# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
# in ~/Downloads/stanford-corenlp-full-2016-10-31

nlp = StanfordCoreNLP('http://localhost:9000')
text = (
    'Pusheen and Smitha walked along the beach. Pusheen wanted to surf,'
    'but fell off the surfboard.')
output = nlp.annotate(text, properties={
    'annotators': 'tokenize,ssplit,pos,depparse,parse',
    'outputFormat': 'json'})
output['sentences'][0]['parse']
output
