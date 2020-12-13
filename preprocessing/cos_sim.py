#https://github.com/egirgin/cmpe493-term-project
#https://towardsdatascience.com/tf-idf-for-document-ranking-from-scratch-in-python-on-real-world-dataset-796d339a4089
#https://stackabuse.com/reading-and-writing-lists-to-a-file-in-python/
import os
import string
import json
import requests

import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

path = "./data/text/raw_data.csv"
stop_words = stopwords.words('english')
punctuations = list(string.punctuation)
data = pd.read_csv(path,encoding="utf-8")

def preprocess(sentence):
    """
      Processes a string
    1-) Replace ' with whitespace
    2-) Replace punctuation with whitespace
    3-) Tokenize
    4-) Stopword removal
    5-) Dismiss token if whitespace or contains a digit
    6-) Lowercase
    7-) Apply Stemming

    :sentence: a string which is the concatenated version of either 'title and abstract' or 'query, question, and narrative'
    :return: a list of tokens the procedures above are applied
    """

    sentence = "" if type(sentence) != type("str") else sentence.replace("'"," ")
    for ch in punctuations:
      sentence = sentence.replace(ch," ")
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not any(ch.isdigit() for ch in w) and len(w) > 1] #and not w in punctuations
    lemmas = [stemmer.stem(word) for word in filtered_sentence]
    return lemmas

def preprocess_lemma(sentence):
    """
      Processes a string
    1-) Replace ' with whitespace
    2-) Replace punctuation with whitespace
    3-) Tokenize
    4-) Stopword removal
    5-) Dismiss token if whitespace or contains a digit
    6-) Lowercase
    7-) Apply Lemmatization

    :sentence: a string which is the concatenated version of either 'title and abstract' or 'query, question, and narrative'
    :return: a list of tokens the procedures above are applied
    """

    sentence = "" if type(sentence) != type("str") else sentence.replace("'"," ")
    for ch in punctuations:
      sentence = sentence.replace(ch," ")
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not any(ch.isdigit() for ch in w) and len(w) > 1] #and not w in punctuations
    lemmas = [lemmatizer.lemmatize(word) for word in filtered_sentence]
    return lemmas

def process_df(data):
    """
      Concat title and abstract then 'preprocess'

      :data: a dataframe with documents in rows; id, title, and abstract in columns
      :return: two lists; first is list of ids (of documents) and the second is 2d list of tokens (rows: docs, columns: tokens)
    """
    lst_index = []
    lst_words = []
    for index, row in data.iterrows():
        tmp = preprocess(row["title"]) + preprocess(row["abstract"])
        lst_words.append(tmp)
        lst_index.append(row["cord_uid"])

        if index%3333 == 0:
          print("% {}".format(int(index*100/len(data))))
    return lst_index, lst_words # lst_words is 2d array -> dim1: document, dim2:tokens in doc

def process_query(data):
    """
      Concat query, question, and narrative then 'preprocess'

      :data: a dataframe with queries in rows; query, question, and narrative in columns
      :return: 2d list of tokens (rows: queries, columns: tokens)
    """
    lst_index = []
    lst_words = []
    for index, row in data.iterrows():
        tmp = preprocess(row["query"] +" "+ row["question"]+ " "+row["narrative"])
        lst_words.append(tmp)
        lst_index.append(row["number"])
    return lst_words # lst_words is 2d array -> dim1: query, dim2:tokens in query


def get_tf_idf_query_similarity(vectorizer, docs_tfidf, query):
    """
      Calculate the cosine similarity between a query vector and all document vectors

      :vectorizer: sklearn's tfidfvectorizer
      :docs_tfidf: vectorized docs (rows: docs, columns: tfidf of tokens)
      :query: a single row of a dataframe
    """

    query_tfidf = vectorizer.transform([" ".join(x) for x in process_query(query)])
    # query_tfidf -> rows: queries (only 1), columns: tfidf value of tokens of query

    cosineSimilarities = cosine_similarity(query_tfidf, docs_tfidf).flatten()
    # cosineSimilarities -> 1 row -> columns are cosine similarity between the query and the corresponding doc

    return cosineSimilarities

 
def main(): 

    # Create tokens of data
    lst_index, lst_words = process_df(data)

    vectorizer = TfidfVectorizer()

    # Create tf-idf vectors
    X = vectorizer.fit_transform([" ".join(x) for x in lst_words])
    # X is a 2d array -> rows: documents, colunms: tf-idf of tokens

    queries = pd.read_json("./data/queries/queries.json")

    # Odd number of queries are training set
    learning_queries = queries.loc[queries.loc[:,"number"] % 2 != 0,:]


    # Calculate cosine similartiy between each doc, query pair

    similarities = []
    for i in range(learning_queries.shape[0]):
        results = get_tf_idf_query_similarity(vectorizer,X,pd.DataFrame(learning_queries.iloc[i,:]).T)
        # cosine similarity between the query and the corresponding doc
        lst = []
        for i in range(len(results)):
            lst.append(results[i])
        similarities.append(lst)

    # similarities(list) -> rows: queries(training_set), columns: cosine similarity to the doc

    similarity_matrix = pd.DataFrame(similarities, index=range(len(learning_queries)), columns=lst_index)

    similarity_matrix.to_csv("./preprocessing/cosine_similarity_matrix.csv")

   
if __name__=="__main__": 
    main() 









    