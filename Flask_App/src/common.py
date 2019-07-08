#!/usr/bin/env python
# coding: utf-8
import re
from nltk.stem.lancaster import LancasterStemmer
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np

def replace_all(text, rep):  
    for i, j in rep.items():  
        text = text.replace(i, j)  
    return text 


def processing(data):
    #Lower
    lower = "".join(data).lower()
    #Remove urls
    no_urls = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '', lower)
    # Remove numeric digits    
    no_nums = re.sub(r'\w*\d\w*', '', no_urls)
    # Remove back slashes
    no_slashes = re.sub(r"\\\\", "", no_nums)
    # Remove ampersand
    no_amps = re.sub("&amp;", "", no_slashes)
    # Expand contractions and abbreviations
    rep = {"isn't":"is not", "can't":"cannot", "don't":"do not", "doesn't":"does not",
           "didn't":"did not", "haven't":"have not", "hadn't":"had not", "hasn't":"has not",
           "i'm":"i am", "you're":"you are", "weren't":"were not", "wasn't":"was not",
           "aren't":"are not", "wouldn't":"would not", "shouldn't":"should not",
           "won't":"will not", "couldn't":"could not", "it's":"it is", "what's":"what is",
           "that's":"that is", "there's":"there is", "admin":"administration", "admins":"administration",
           "alot":"a lot", "i.e.":"ie", "customer service": "customer support",
           "feel like":"feel as if", "foh":"front of house", "front of the house":"front of house",
           "boh":"back of house", "back of the house":"back of house"}
    text = replace_all(str(no_amps), rep)
    #Manually separate sentences
    sep_sents = re.sub(r'(?<=[.])(?=[^\s])', ' ', text)
    # Correct Mistakes
    blob = TextBlob(sep_sents).correct()
    # Stem
    stemmer = LancasterStemmer()
    stem = ''.join([stemmer.stem(word) for word in blob])
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    lem = ''.join([lemmatizer.lemmatize(word) for word in stem])
    # Replace known words, tht were likely autocorrected
    rep2 = {"nor":"ncr", "but":". But", "pus":"pos", "unfordable":"unaffordable"}
    text2 = replace_all(lem, rep2)
    return(text2)


def group_by_sentiment(vectorized_df, sentiment_col, sentiment_type):
    vectorized_df['sentiment'] = sentiment_col
    return vectorized_df[(vectorized_df['sentiment'] == str(sentiment_type))].drop(['sentiment'], axis=1)
    

def pickle_stuff(pickle_list):
    for name, value in pickle_list.items():
        filepath = '/Users/tim/src/Metis/Project_4/data/interim/' + str(name) + '.pkl'
        with open(filepath, 'wb') as pkl:
            pickle.dump(value, pkl)

def open_pickles(pickle_names):
    pickles = {}
    for name in pickle_names:
        filepath = '/Users/tim/src/Metis/Project_4/data/interim/' + str(name) + '.pkl'
        with open(filepath, 'rb') as pkl:
            pickles[name] = pickle.load(pkl)
    return pickles

def model_nmf(vectorized_df, categories):
    index = []
    for category in range(1,categories+1):
        index.append('component_' + str(category))
    nmf_model = NMF(categories)
    doc_topic = nmf_model.fit_transform(vectorized_df)
    topic_word = pd.DataFrame(nmf_model.components_.round(3),
             index = index,         
             columns = vectorized_df.columns)
    return doc_topic

def model_lsa(vectorized_df, categories):
    index = []
    for category in range(1,categories+1):
        index.append('component_' + str(category))
    lsa_model = TruncatedSVD(categories)
    doc_topic = lsa_model.fit_transform(vectorized_df)
    topic_word = pd.DataFrame(lsa_model.components_.round(3),
             index = index,         
             columns = vectorized_df.columns)
    return topic_word, lsa_model, doc_topic

def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

def inertia_curve(num1, num2, topic):
    inertia_dict = {}
    for c in list(range(num1,num2+1)):
        km = KMeans(n_clusters=c)
        km.fit(topic)
        inertia = km.inertia_
        inertia_dict[c] = inertia
    return inertia_dict

def display_cluster(X,model,num_clusters):
    color = 'brgcmyk'
    alpha = 0.5
    s = 20
    if num_clusters == 0:
        plt.scatter(X[:,0],X[:,1],c = color[0],alpha = alpha,s = s)
    else:
        for i in range(num_clusters):
            plt.scatter(X[model.labels_==i,0],X[model.labels_==i,1],c = color[i],alpha = alpha,s=s)
            plt.scatter(model.cluster_centers_[i][0],model.cluster_centers_[i][1],c = color[i], marker = 'x', s = 100)
            

def ClusterIndicesNumpy(clustNum, labels_array): #numpy 
    return np.where(labels_array == clustNum)[0]
            
