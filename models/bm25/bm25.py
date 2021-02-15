from rank_bm25 import BM25L
from rank_bm25 import BM25Plus
from rank_bm25 import BM25Okapi

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

path = "./drive/MyDrive/data/text/raw_data.csv"
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
        if index == 1327 or index == 1328:
          print(row["title"])
          print(row["abstract"])
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



# Create tokens of data
lst_index, lst_words = process_df(data)

# Create models from tokenized corpus
bm25 = BM25Okapi(lst_words)
bm25L = BM25L(lst_words)
bm25Plus = BM25Plus(lst_words)

queries = pd.read_json("./drive/MyDrive/data/queries/queries.json")

# Preprocess and tokenize queries
queryList = {}
# queries = testing_queries
# numbers = range(1,50,2)
numbers = range(0,50)
for i in queries:
    if i == "number":
        continue
    for j in numbers:
        if j not in queryList:
            queryList[j] = []
        queryList[j].extend(preprocess(queries[i][j]))

# Get the scores of documents for each query
scores = []
scoresL = []
scoresPlus = []
for q in queryList:
  scores.append(bm25.get_scores(queryList[q]))
  scoresL.append(bm25L.get_scores(queryList[q]))
  scoresPlus.append(bm25Plus.get_scores(queryList[q]))

okapi_matrix = pd.DataFrame(scores, index=range(len(queries)), columns=lst_index)

okapi_matrix_L = pd.DataFrame(scores, index=range(len(queries)), columns=lst_index)

okapi_matrix_Plus = pd.DataFrame(scores, index=range(len(queries)), columns=lst_index)

# dump scores for eval_bm25.py
okapi_matrix.to_csv("okapi_matrix.csv")
okapi_matrix_L.to_csv("okapi_matrix_L.csv")
okapi_matrix_Plus.to_csv("okapi_matrix_Plus.csv")
