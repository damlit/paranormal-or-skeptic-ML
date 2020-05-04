#!/usr/bin/env python
import sys
import pickle
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


# xzcat in.tsv.xz | paste expected.tsv - | ./train.py


def train():
    tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')
    documents_total = 0
    sceptic_documents_total = 0
    paranormal_documents_total = 0

    vocabulary = set()
    sceptic_words_total = 0
    paranormal_words_total = 0

    paranormal_words = defaultdict(int)
    sceptic_words = defaultdict(int)

    for line in sys.stdin:
        line = line.rstrip()
        line = line.split("\t")
        label = line[0].strip()
        document = line[1]
        terms = tokenizer.tokenize(document)

        for term in terms:
            vocabulary.add(term)

        documents_total += 1
        if label == 'S':
            sceptic_documents_total += 1
            sceptic_words_total += len(terms)
            for term in terms:
                sceptic_words[term] += 1
        else:
            paranormal_documents_total += 1
            paranormal_words_total += len(terms)
            for term in terms:
                paranormal_words[term] += 1

    sceptic = sceptic_documents_total / documents_total
    vocabulary_size = len(vocabulary)

    model = (sceptic, vocabulary_size, sceptic_words_total, paranormal_words_total, sceptic_words, paranormal_words)
    with open('./bayes_model.pkl', 'wb') as pickle_file:
        pickle.dump(model, pickle_file)


if __name__ == '__main__':
    train()