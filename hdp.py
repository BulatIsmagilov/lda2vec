# Run in python console
import nltk;

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, HdpModel

# spacy for lemmatization
import spacy

from IPython import embed
import parse_docs
import generate_corpus

texts, id2word, corpus = generate_corpus.generate_corpus()

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

hdpmodel = HdpModel(corpus=corpus, id2word=id2word)

hdpmodel.show_topics()

coherence_model_hdp = CoherenceModel(model=hdpmodel, texts=texts, dictionary=id2word, coherence='c_v')

coherence_hdp = coherence_model_hdp.get_coherence()
print('\nCoherence Score: ', coherence_hdp)
