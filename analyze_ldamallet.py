import gensim
import numpy as np
import pandas as pd
from IPython import embed

from ggplot import *
import json
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pyLDAvis.gensim
import seaborn as sns
import warnings
import pickle
import model_visualization
from gensim.models import CoherenceModel, HdpModel

# %matplotlib inline
# pyLDAvis.enable_notebook()
# %matplotlib inline
# pyLDAvis.enable_notebook()
'exec(%matplotlib inline)'

#pd.options.display.max_rows = 10

# with open('text_corpus_id2word_fulldata.pkl', 'rb') as handle:
#    dictionary, id2word, corpus, full_data = pickle.load(handle)
#
# mallet_path = '/Users/ismglv/dev/lda2vec/mallet/bin/mallet' # update this path
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=60, id2word=id2word)
# ldamallet.save('lda.model')
# ldamallet = gensim.models.wrappers.LdaMallet.load('ldamallet.model')
# model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)

# pyLDAvis.gensim.prepare(model, corpus, id2word)

with open('text_corpus_id2word_fulldata.pkl', 'rb') as handle:
   dictionary, id2word, corpus, full_data = pickle.load(handle)


#mallet_path = '/Users/ismglv/dev/lda2vec/mallet/bin/mallet' # update this path
#ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=60, id2word=id2word)
#coherence_model = CoherenceModel(model=ldamallet, texts=dictionary, dictionary=id2word, coherence='c_v')
#coherence = coherence_model.get_coherence()
#print('\nCoherence Score: ', coherence)
#ldamallet.save('lda.model')

ldamallet =  gensim.models.wrappers.LdaMallet.load('lda.model')
model = gensim.models.wrappers.ldamallet.malletmodel2ldamodel(ldamallet)
df_topic_sents_keywords = model_visualization.format_topics_sentences(ldamallet, corpus, dictionary)

df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['doc_id', 'dom_topic', 'topic_weight', 'topic_words', 'Text']



# to get doc topics df_dominant_topic['Keywords'][doc_num]
# to get doc topics dominant quality df_dominant_topic['Dominant_Topic'][doc_num]

#Find the most representative document for each topic

# Number of Documents for Each Topic
topic_counts = df_topic_sents_keywords['dom_topic'].value_counts()

# Percentage of Documents for Each Topic
topic_contribution = round(topic_counts/topic_counts.sum(), 4)

# Topic Number and Keywords
topic_num_keywords = df_topic_sents_keywords[['dom_topic', 'topic_words']]

# Concatenate Column wise
df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)

# Change Column names
df_dominant_topics.columns = ['dom_topic', 'topic_words', 'Num_Documents', 'Perc_Documents']

# Show
df_dominant_topics

data = pd.merge(full_data, df_dominant_topic, left_index=True, right_index=True)

import operator
from functools import reduce
print(data.columns)
print(df_dominant_topic.columns)
data.topic_words_array = data.topic_words.apply(lambda x: list(map(lambda y: y.strip(), x.split(','))))
order = set(reduce(operator.concat, data.topic_words_array))
print(order)

embed()
