import os
import glob
import errno
from IPython import embed
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.utils import simple_preprocess
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import re

import numpy as np
import pandas as pd
import meta_parsing

import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

path = "/Users/ismglv/dev/news-topics/solidopinion_brexit/documents/*.txt"
files = glob.glob(path)

stop_words = stopwords.words('english')

metadata = meta_parsing.metadata()

def read_files():
    texts = []
    for name in files:
        try:
            with open(name) as f:
                texts.append([name.split('/')[-1], prepare_text_for_lda(f.read())])
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    texts = pd.DataFrame(texts, columns=["filename", "words"])
    texts = pd.concat([texts, metadata])
    return texts


def remove_stopwords(text, stop_words):
    clear_text = []
    for word in simple_preprocess(text):
        if word not in stop_words:
            clear_text.append(word)
    return ' '.join(map(str, clear_text))


def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

def prepare_text_for_lda(text):
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\'", "", text)

    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)
