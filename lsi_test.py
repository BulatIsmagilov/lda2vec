# Run in python console
import nltk;

import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, HdpModel
# spacy for lemmatization
import spacy

from IPython import embed
import parse_docs
import generate_corpus
import model_visualization

import pickle

with open('text_corpus_id2word_fulldata.pkl', 'rb') as handle:
   texts, id2word, corpus, full_data = pickle.load(handle)
# texts, id2word, corpus = generate_corpus.generate_corpus()

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


tfidf = gensim.models.TfidfModel(corpus)  # step 1 -- initialize a model
tfidf_corpus = tfidf[corpus]

vectors_path = "/Users/ismglv/dev/lda2vec/GoogleNews-vectors-negative300.bin"
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(vectors_path, binary=True)

for i in [50, 100, 150]:
    lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=id2word, num_topics=i)

    # coherence_model = CoherenceModel(model=lsi_model, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_model_cw2v = CoherenceModel(model=lsi_model, texts=texts, dictionary=id2word, coherence='c_w2v', keyed_vectors=keyed_vectors)
    # coherence_model_cnmpi = CoherenceModel(model=lsi_model, texts=texts, dictionary=id2word, coherence='c_npmi')
    # coherence_model_cuci = CoherenceModel(model=lsi_model, texts=texts, dictionary=id2word, coherence='c_uci')
    # coherence_model_u_mass = CoherenceModel(model=lsi_model, texts=texts, dictionary=id2word, coherence='u_mass')

    # coherence = coherence_model.get_coherence()
    coherence_cw2v = coherence_model_cw2v.get_coherence()
    # coherence_cnmpi = coherence_model_cnmpi.get_coherence()
    # coherence_cuci = coherence_model_cuci.get_coherence()
    # coherence_umass = coherence_model_u_mass.get_coherence()
    # print('\nCoherence Score: ', coherence)
    print('\nCoherence C_W2V Score: ', coherence_cw2v)
    # print('\nCoherence C_NMPI Score: ', coherence_cnmpi)
    # print('\nCoherence C_UCI Score: ', coherence_cuci)
    # print('\nCoherence U_MASS Score: ', coherence_umass)
