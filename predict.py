#!/usr/bin/env python
import sys
import pickle
import math
from nltk.tokenize import RegexpTokenizer


# xzcat dev-0/in.tsv.xz | python3 ./predict.py > dev-0/out.tsv


psceptic, vocabulary_size, sceptic_words_total, paranormal_words_total, sceptic_words, paranormal_words = \
    pickle.load(open('bayes_model.pkl', 'rb'))
tokenizer = RegexpTokenizer(r'\b[^\d\W]+\b')

for line in sys.stdin:
    document = line.rstrip()
    terms = tokenizer.tokenize(document)
    log_prob_sceptic = math.log(psceptic)
    log_prob_paranormal = math.log(1 - psceptic)

    for term in terms:
        if term not in sceptic_words:
            sceptic_words[term] = 0
        if term not in paranormal_words:
            paranormal_words[term] = 0

        log_prob_sceptic += math.log((sceptic_words[term] + 1) / (sceptic_words_total + vocabulary_size))
        log_prob_paranormal += math.log((paranormal_words[term] + 1) / (paranormal_words_total + vocabulary_size))

    if log_prob_sceptic > log_prob_paranormal:
        print(" S")
    else:
        print(" P")