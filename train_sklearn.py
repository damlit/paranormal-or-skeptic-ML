#!/usr/bin/env python
import sys
import pickle
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import RegexpTokenizer


# xzcat in.tsv.xz | paste expected.tsv - | ./train.py


def train():

    tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
    label_list = list()
    document_list = list()

    for line in sys.stdin:
        line = line.rstrip()
        line = line.split("\t")
        label = line[0].strip()
        document = line[1]
        label_list.append(label)
        document_list.append(document)

    documents = pd.DataFrame({'label': label_list, 'text': document_list})

    documents['label'] = documents.label.map({'P': 0, 'S': 1})
    documents['text'] = documents.text.map(lambda x: x.lower())
    documents['text'] = documents.text.str.replace('[^\w\s]', '')
    documents['text'] = documents['text'].apply(tokenizer.tokenize)

    stemmer = PorterStemmer()
    documents['text'] = documents['text'].apply(lambda x: [stemmer.stem(y) for y in x])

    documents['text'] = documents['text'].apply(lambda x: ' '.join(x))
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(documents['text'])

    transformer = TfidfTransformer().fit(counts)
    counts = transformer.transform(counts)

    X_train, X_test, y_train, y_test = train_test_split(counts, documents['label'], test_size=0.001, random_state=69)

    model = MultinomialNB().fit(X_train, y_train)

    with open('./bayes_model_sk.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)
    with open('./transformer.pkl', 'wb') as pickle_file:
        pickle.dump(transformer, pickle_file)
    with open('./countVec.pkl', 'wb') as pickle_file:
        pickle.dump(count_vect, pickle_file)


if __name__ == '__main__':
    train()