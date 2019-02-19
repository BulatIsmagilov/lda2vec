import os
import glob
import errno
from IPython import embed
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from gensim.utils import simple_preprocess

path = "/Users/ismglv/dev/news-topics/solidopinion_brexit/documents/*.txt"
files = glob.glob(path)

stop_words = stopwords.words('english')

def read_files():
    texts = []
    for name in files[0:100]:
        try:
            with open(name) as f:
                texts.append(remove_stopwords(f.read(), stop_words))
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    return texts


def remove_stopwords(text, stop_words):
    clear_text = []
    for word in simple_preprocess(text):
        if word not in stop_words:
            clear_text.append(word)
    return ' '.join(map(str, clear_text))

