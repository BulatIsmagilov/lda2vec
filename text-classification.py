from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances
from pprint import pprint

import pandas, xgboost, numpy, textblob, string, spacy, gensim
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers

from IPython import embed
import parse_docs

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'

# load the dataset
data = parse_docs.read_files()
labels, texts = [], []
for text in data:
    labels.append("brexit")
    texts.append(text)

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

# label encode the target variable
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

# train a LDA Model
lda = LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=100)
X_topics = lda.fit_transform(xtrain_count)
topic_word = lda.components_
vocab = count_vect.get_feature_names()

# view the topic models
n_top_words = 5
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

print(topic_summaries)

# pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, xtrain_count, count_vect, mds='tsne')
panel

nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations


df_topic_keywords = pandas.DataFrame(topic_summaries)

def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = count_vect.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = lda.transform(mytext_4)
    topic = df_topic_keywords.iloc[numpy.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
mytext = ["strasbourg stphane strain strangling straight"]
topic, prob_scores = predict_topic(text = mytext)
print(topic)

def similar_documents(text, doc_topic_probs, documents = data, nlp=nlp, top_n=5, verbose=False):
    topic, x  = predict_topic(text)
    dists = euclidean_distances(x.reshape(1, -1), doc_topic_probs)[0]
    doc_ids = numpy.argsort(dists)[:top_n]
    if verbose:
        print("Topic KeyWords: ", topic)
        print("Topic Prob Scores of text: ", numpy.round(x, 1))
        print("Most Similar Doc's Probs:  ", numpy.round(doc_topic_probs[doc_ids], 1))
    return doc_ids, numpy.take(documents, doc_ids)

# Get similar documents
mytext = ["strasbourg stphane strain strangling straight"]
doc_ids, docs = similar_documents(text=mytext, doc_topic_probs=X_topics, documents = data, top_n=1, verbose=True)
print('\n', docs[0][:500])
