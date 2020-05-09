# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:13:04 2020

@author: rahul
"""

import re
import nltk
import pandas as pd
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy.sparse import coo_matrix

stop_words = set(stopwords.words("english"))

def data_process(dataset):
    corpus = []
    text = re.sub('[^a-zA-Z]', ' ', dataset)
    # text = text.lower()
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    text = re.sub("(\\d|\\W)+"," ",text)
    text = text.split()
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
          stop_words] 
    text = " ".join(text)
    return text

def tfidf_Data (data,m, n,x,y):
    """
    data: problem statement
    m = document to pass though the tfidf transform
    n = countvectorizer features (integer)
    (x,y) = ngram range
    
    """
    corpus = data_process(data)
    corpus = [corpus]
    cv=CountVectorizer(stop_words=stop_words, max_features=n, ngram_range=(x,y))
    X=cv.fit_transform(corpus)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    feature_names=cv.get_feature_names()
    doc=corpus[m]
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
    return tf_idf_vector,feature_names

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


dataset = '''
Traditional electronics are connected with wires to the power supply to switches and sensors. Over the time the amount of sensors and IoT devices is growing so there is no more space in future for all the cables needed for the connections.
MID (molded interconnect devices) opens the possibility to place tracks directly on the surface of plastic parts.
MID must be further developed for our own applications. E.g. printing in plastic housing parts.
We expect that this will be the enabler for much more sensors in our tools. 5 years ago the MID technology was not competitive vs. classical cabling/wiring. With increasing electronics, cabling space gets rare. What technologies and companies exist on the market to create MID? Interesting technologies could be laser activation, galvanization, deformable sheets.
'''

KE, feature_names = tfidf_Data(dataset, 0, 1000, 1,3)
sorted_items = sort_coo(KE.tocoo())
KE = extract_topn_from_vector(feature_names,sorted_items,10)
print(KE)