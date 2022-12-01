# RUN as:
# python 01-train.py 3 2 cyrl-latn-aligned-words.tsv cyrl-latn-3-2.model

import argparse
from joblib import dump
import random
import re

import numpy as np
from sklearn import tree
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

RANDOM_STATE=0
random.seed(RANDOM_STATE)

def get_alignments(filename):
    X, y = [], []
    with open(filename, "r") as inf:
        for line in inf.readlines():
            cyrl, latn = line.split('\t')
            X.append(cyrl.strip())
            y.append(latn.strip())
    return X, y

def extract_features(word: str, before_pad_length=0, after_pad_length=0):
    features = []
    before_pad = "∅" * before_pad_length
    after_pad = "∅" * after_pad_length
    word = before_pad + word + after_pad
    for i in range(before_pad_length, len(word) - after_pad_length):
        features.append(list(word[i-before_pad_length:i+1+after_pad_length]))
    return features


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def train(alignment_file, before_pad_length, after_pad_length):
    X, Y = get_alignments(alignment_file)
    X = flatten([extract_features(''.join(x.split('|')), before_pad_length, after_pad_length)
         for x in X])
    Y = flatten([y.split('|') for y in Y])

    enc = OneHotEncoder(sparse=False)
    X = enc.fit_transform(X)

    model = tree.DecisionTreeClassifier()
    model.fit(X, Y)

    model.enc = enc
    model.before_pad_length = before_pad_length
    model.after_pad_length = after_pad_length

    return model


def save_model(model, path):
    dump(model, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a Decision Tree classifier.')
    parser.add_argument('before_pad_length', type=int)
    parser.add_argument('after_pad_length', type=int)
    parser.add_argument('alignment_file',
                        help="TSV where first column is Cyrillic words and " +
                        "the second column is the Latin words. Words are " +
                        "aligned with pipes.")
    parser.add_argument('model_save_path',
                        help="Where to save the output model.")
    args = parser.parse_args()
    model = train(args.alignment_file, args.before_pad_length,
                       args.after_pad_length)
    save_model(model, args.model_save_path)
