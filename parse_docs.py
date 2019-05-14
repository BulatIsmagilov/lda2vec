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
                texts.append([name.split('/')[-1], tokenize(f.read())])
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    texts = pd.DataFrame(texts, columns=["filename", "words"])
    return texts.merge(metadata, left_on='filename', right_on='filename')

def tokenize(text):
    return parser(text)
