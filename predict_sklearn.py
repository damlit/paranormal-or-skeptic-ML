#!/usr/bin/env python
import sys
import pickle
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
model = pickle.load(open('bayes_model_sk.pkl', 'rb'))
transformer = pickle.load(open('transformer.pkl', 'rb'))
count_vec = pickle.load(open('countVec.pkl', 'rb'))
document_list = list()

for line in sys.stdin:
    line = line.rstrip()
    line = line.split("\t")
    document = line[0]
    document_list.append(document)

documents_to_predict = pd.DataFrame({'text': document_list})
documents_to_predict['text'] = documents_to_predict.text.map(lambda x: x.lower())
documents_to_predict['text'] = documents_to_predict.text.str.replace('[^\w\s]', '')
documents_to_predict['text'] = documents_to_predict['text'].apply(tokenizer.tokenize)

stemmer = PorterStemmer()
documents_to_predict['text'] = documents_to_predict['text'].apply(lambda x: [stemmer.stem(y) for y in x])

documents_to_predict['text'] = documents_to_predict['text'].apply(lambda x: ' '.join(x))

counts = count_vec.transform(documents_to_predict['text'])

counts = transformer.transform(counts)

predict = model.predict(counts)

for pr in predict:
    if (pr == 0):
        print(" P")
    else:
        print(" S")