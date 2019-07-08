import pickle
import pandas as pd

import nltk
from nltk import sent_tokenize
from nltk.tokenize import MWETokenizer
from nltk.corpus import stopwords

from sklearn import preprocessing
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from common import processing, display_topics, model_nmf, ClusterIndicesNumpy

def find_clusters(data):
    # Process Text
    processed_text = processing(data)

    # Set noun phrases
    mwe_tokenizer = MWETokenizer([('customer','service'), ('hard', 'to'), ('service', 'calls'),
                             ('over', 'seas'), ('follows', 'up'), ('user', 'friendly'), ('long','time'), 
                             ('front', 'of', 'house'), ('back', 'of', 'house'), ('behind', 'the', 'times'),
                             ('out', 'of', 'date'), ('easy', 'to')])

    # Tokenize by sentence
    sent_token = mwe_tokenizer.tokenize(sent_tokenize(processed_text))

    # Customize stop words 
    new_list= ('pos', 'product', 'system', 'software', 'hardware', 'program')
    stopset = set(nltk.corpus.stopwords.words('english'))
    stopset.update(new_list)

    # Vectorize data
    tf = TfidfVectorizer(stop_words=stopset)
    x_tf = tf.fit_transform(sent_token).toarray()
    tf_vector = pd.DataFrame(x_tf,columns=tf.get_feature_names())

    # Model
    #nmf_model, 
    doc_topic = model_nmf(tf_vector, 5)

    # Normalize
    temp = pd.DataFrame(doc_topic)
    print("temp")
    print(temp)
    temp2 = preprocessing.normalize(temp, norm='l1', axis=1, copy=True, return_norm=False)


    # Cluster
    num_clusters = 5
    km = KMeans(n_clusters=num_clusters,random_state=10)
    km.fit(temp2)

    # Get cluster
    review_dict = {}

    for num in range(1, num_clusters + 1):
        # Get points
        review_list = []
        for i,j in enumerate(ClusterIndicesNumpy(num, km.labels_)):
            #Append points to list
            review_list.append(str(i +1) +"." + sent_token[j])
        # Match list to cluster    
        review_dict["Category " + str(num)] = review_list

    # Remove tokens that are less than 15 characters long
    # result = {} 
    # for num in range(1, num_clusters+1):    
    #     result_list = []
    #     for item in review_dict['Category_' + str(num)]:
    #         if len(j) >=15:
    #             result_list.append(item)
    #             result["Category " + str(num)] =  item
    
    return review_dict

if __name__ == '__main__':
    pass