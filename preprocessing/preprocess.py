import os
import string
import json
import requests
from tqdm import tqdm
import os.path
from os import path
import pickle
import argparse

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

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument('-l',
                        '--lemmatization',
                        action='store_true')

args = arg_parser.parse_args()


data_path = "./data/text/raw_data.csv"
stop_words = stopwords.words('english')
punctuations = list(string.punctuation)
data = pd.read_csv(data_path,encoding="utf-8")


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

def process_df(data, lemma = False):
    """
      Concat title and abstract then 'preprocess'

      :data: a dataframe with documents in rows; id, title, and abstract in columns
      :return: two lists; first is list of ids (of documents) and the second is 2d list of tokens (rows: docs, columns: tokens)
    """
    if lemma:
        print("Lemmatization is being used.")

    lst_index = []
    lst_words = []
    for index, row in tqdm(data.iterrows(), total=len(data)):
        if lemma:
            tmp = preprocess_lemma(row["title"]) + preprocess_lemma(row["abstract"])   
        else: 
            tmp = preprocess(row["title"]) + preprocess(row["abstract"])
        lst_words.append(tmp)
        lst_index.append(row["cord_uid"])

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


def main():

    print("Creating Corpus...")
    lst_index, lst_words = process_df(data, args.lemmatization)
    with open("./preprocessing/processed_data.pickle", "wb") as processedData:
        pickle.dump((lst_index, lst_words), processedData)

    queries = pd.read_json("./data/queries/queries.json")

    #######################################################################
    # Odd number of queries are training set
    learning_queries = queries.loc[queries.loc[:,"number"] % 2 != 0,:]
    processed_train_queries = []

    for i in range(learning_queries.shape[0]):
        query = process_query(pd.DataFrame(learning_queries.iloc[i,:]).T)
        processed_train_queries.append(query)

    with open("./preprocessing/training_queries.pickle", "wb") as processedData:
        pickle.dump(processed_train_queries, processedData)

    #######################################################################

    testing_queries = queries.loc[queries.loc[:,"number"] % 2 == 0,:]
    processed_test_queries = []

    for i in range(testing_queries.shape[0]):
        query = process_query(pd.DataFrame(testing_queries.iloc[i,:]).T)
        processed_test_queries.append(query)

    with open("./preprocessing/testing_queries.pickle", "wb") as processedData:
        pickle.dump(processed_test_queries, processedData)


if __name__=="__main__": 
    main() 
