# Run in python console
# import nltk;
import os
import re
import operator
import matplotlib.pyplot as plt
import warnings
import gensim
import numpy as np
import pandas as pd
# warnings.filterwarnings('ignore')  # Let's not pay heed to them right now

from gensim.models import CoherenceModel, HdpModel
# spacy for lemmatization
import spacy

from IPython import embed
# import parse_docs
# import generate_corpus
import model_visualization
import pickle

with open('text_corpus_id2word_fulldata.pkl', 'rb') as handle:
   texts, id2word, corpus, full_data = pickle.load(handle)
#  texts, id2word, corpus, full_data = generate_corpus.generate_corpus()
# embed()

# View
print(corpus[:1])

# Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]


mallet_path = '/Users/ismglv/dev/lda2vec/mallet/bin/mallet' # update this path
# ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=60, id2word=id2word)
# coherence_model = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
# coherence = coherence_model.get_coherence()
# print('\nCoherence Score: ', coherence)

vectors_path = "/Users/ismglv/dev/lda2vec/GoogleNews-vectors-negative300.bin"
keyed_vectors = gensim.models.KeyedVectors.load_word2vec_format(vectors_path, binary=True)

for i in [50, 100, 150]:
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=i, id2word=id2word)

    # coherence_model_hdp = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')

    # coherence_hdp = coherence_model_hdp.get_coherence()
    # print('\nCoherence Score: ', coherence_hdp, i)

    coherence_model_cw2v = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_w2v', keyed_vectors=keyed_vectors)
    # coherence_model_cnmpi = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_npmi')
    # coherence_model_cuci = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_uci')
    # coherence_model_u_mass = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='u_mass')

    # # coherence = coherence_model.get_coherence()
    coherence_cw2v = coherence_model_cw2v.get_coherence()
    # coherence_cnmpi = coherence_model_cnmpi.get_coherence()
    # coherence_cuci = coherence_model_cuci.get_coherence()
    # coherence_umass = coherence_model_u_mass.get_coherence()
    # # print('\nCoherence Score: ', coherence)
    print('\nCoherence C_W2V Score: ', coherence_cw2v, i)
    # print('\nCoherence C_NMPI Score: ', coherence_cnmpi, i)
    # print('\nCoherence C_UCI Score: ', coherence_cuci, i)
    # print('\nCoherence U_MASS Score: ', coherence_umass, i)



# df_topic_sents_keywords = model_visualization.format_topics_sentences(ldamallet, corpus, texts)
#
# df_dominant_topic = df_topic_sents_keywords.reset_index()
# df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
#
#
#
# # to get doc topics df_dominant_topic['Keywords'][doc_num]
# # to get doc topics dominant quality df_dominant_topic['Dominant_Topic'][doc_num]
#
# #Find the most representative document for each topic
#
# # Number of Documents for Each Topic
# topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
#
# # Percentage of Documents for Each Topic
# topic_contribution = round(topic_counts/topic_counts.sum(), 4)
#
# # Topic Number and Keywords
# topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
#
# # Concatenate Column wise
# df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
#
# # Change Column names
# df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents', 'Perc_Documents']
#
# # Show
# df_dominant_topics
#
# print(topic_num_keywords.Topic_Keywords.value_counts())
