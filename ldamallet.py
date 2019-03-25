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


texts, id2word, corpus = generate_corpus.generate_corpus()

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


mallet_path = '/Users/ismglv/dev/lda2vec/mallet-2.0.8/bin/mallet' # update this path
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=100, id2word=id2word)

coherence_model_hdp = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')

coherence_hdp = coherence_model_hdp.get_coherence()
print('\nCoherence Score: ', coherence_hdp)

df_topic_sents_keywords = model_visualization.format_topics_sentences(ldamallet, corpus, texts)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']



# to get doc topics df_dominant_topic['Keywords'][doc_num]
# to get doc topics dominant quality df_dominant_topic['Dominant_Topic'][doc_num]

#Find the most representative document for each topic

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

print(topic_num_keywords.Topic_Keywords.value_counts())
