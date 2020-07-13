# -*- coding: utf-8 -*-
import copy
import pandas as pd
import numpy as np
import sys
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import linear_kernel
from stemming.porter2 import stem

#from measure_time import time_func

data_path = 'GSR.json'
results = []    
class SearchEngine():  
    replace_words = {'&': '_and_', 'unknown':' '}    

    def __init__(self, text_column='snippet', title_column='title', domain_column='link', url_column='link'):
        self.text_column = text_column
        self.url_column = url_column
        self.domain_column = domain_column
        self.title_column = title_column 
        pass
    
    '''
    Use the TFIDF transformer for KSE along with parameters:
    - ngram range: select the combination of words
    - perform_stem: True for stemming else False
    '''

    def fit(self, df, ngram_range=(1,3), perform_stem=True):
        self.df = df
        self.perform_stem = perform_stem
        doc_df = self.preprocess(df)
        stopWords = stopwords.words('english')    
        self.vectoriser = CountVectorizer(stop_words = stopWords, ngram_range=ngram_range)
        train_vectorised = self.vectoriser.fit_transform(doc_df)
        self.transformer = TfidfTransformer()
        self.transformer.fit(train_vectorised)
        self.fitted_tfidf = self.transformer.transform(train_vectorised)


    '''
    Data/Query pre-processing
    '''

    def preprocess(self, df):
        result = df[self.text_column]
        result = np.core.defchararray.lower(result.values.astype(str))
        for word in self.replace_words:
            result = np.core.defchararray.replace(result, word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_array(result)
        return result

    def preprocess_query(self, query):
        result = query.lower()
        for word in self.replace_words:
            result = result.replace(word, self.replace_words[word])
        if self.perform_stem:
            result = self.stem_document(result)
        return result

    def stem_array(self, v):
        result = np.array([self.stem_document(document) for document in v])
        return result
    
    '''
    Stemming to be carried to cut down the prefix or suffix of a word
    '''

    def stem_document(self, text):
        #lemma = WordNetLemmatizer()
        result = " ".join([stem(w) for w in nltk.word_tokenize(text)])
        #result = [stem(word) for word in text.split(" ")]
        #result = ' '.join(result)
        return result
    
    '''
    to get the scoring results of the top keywords
    the parameter 
    - max_rows: can be tuned for the number of results.
    '''

    def get_results(self, query, max_rows=10):
        score = self.get_score(query)
        print(score)
        results_df = copy.deepcopy(self.df)
        results_df['Score'] = score
        results_df = results_df.loc[score>0]
        results_df = results_df.iloc[np.argsort(-results_df['Score'].values)]
        results_df = results_df.head(max_rows)
        self.print_results(results_df, query)
        return results_df      
       
    def get_score(self, query):
        query_vectorised = self.vectoriser.transform([query])    
        query_tfidf = self.transformer.transform(query_vectorised)
        cosine_similarities = linear_kernel(self.fitted_tfidf, query_tfidf).flatten()
        return cosine_similarities
        # normalization
    
    def print_results(self, df, query):
        print("---------")
        print('results for "{}"'.format(query))
        print(df)
        df_re_order = df[['Score','title','domain','url','snippet']]  
        df_re_order.to_csv('results_KSE.csv', index = True, header = True)

        """
        for i, row in df.iterrows():
            print('{}, Title : {}, Domain: {}, Url: {}, Description: {}'.format(
                    row['ranking_score'],
                    row[self.title_column],
                    row[self.domain_column],
                    row[self.url_column],
                    row[self.text_column]))
        #my_df = pd.DataFrame(results)
        #my_df.to_csv('results.csv', index=False, header=False)
        """
        
def load_data(path):
    df = pd.read_json(path)
    return df


if __name__ == '__main__':
    queries = [
        'molded interconnect devices'
        ]

    df = load_data(data_path)
    model = SearchEngine(text_column='snippet', title_column='title', domain_column='domain', url_column='url')
    model.fit(df, perform_stem=False)

    for query in queries:
        model.get_results(query)
