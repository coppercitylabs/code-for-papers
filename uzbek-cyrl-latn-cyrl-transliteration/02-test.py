# RUN as:
# python 02-test.py cyrl-latn-3-2.model cyrl-words.txt

import argparse
from joblib import load
import importlib

classifier = importlib.import_module("01-train")


def load_model(model_path):
    return load(model_path)


def load_words(filename):
    with open(filename, "r") as inf:
        return [x.strip() for x in inf.readlines()]


def decode(model, words):
    for word in words:
        # We don't have training data for words starting with an uppercase letter
        if word[0].isupper():
            print('-----------------------')
            print(word)
            word = word.lower()
        features = classifier.extract_features(
            word, model.before_pad_length, model.after_pad_length)
        pred = ''.join(model.predict(model.enc.transform(features)))
        if word.islower():
            pred = pred.lower()
        elif word.isupper():
            pred = pred.upper()
        print(pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transliterate.')
    parser.add_argument('model_path', help="Pretrained model path.")
    parser.add_argument('input_filename',
                        help="Name of the input file that contains words on each line.")
    args = parser.parse_args()
    model = load_model(args.model_path)
    words = load_words(args.input_filename)
    decode(model, words)
