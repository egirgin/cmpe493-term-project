import os
import string
import json
import requests
from tqdm import tqdm

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

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
    for index, row in tqdm(data.iterrows(), total=len(data)):
        tmp = preprocess(row["title"]) + preprocess(row["abstract"])
        lst_words.append(tmp)
        lst_index.append(row["cord_uid"])

        #if index%3333 == 0:
        #  print("% {}".format(int(index*100/len(data))))
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

def list2tagged(doc_list, id_list):
    tagged_documents = []
    
    for doc, uid in zip(doc_list, id_list):
        tagged_documents.append(TaggedDocument(doc, [uid]))

    return tagged_documents


def main():
    print("Creating Corpus...")
    lst_index, lst_words = process_df(data)

    os.system("clear")
    print("Converting Tagged Document...")
    train_corpus = list2tagged(lst_words, lst_index)
    os.system("clear")
    """
      vector_size : doc embedding size
      min_count : discard threshold
      dm : Defines the training algorithm. If `dm=1`, 'distributed memory' (PV-DM) is used.
            Otherwise, `distributed bag of words` (PV-DBOW) is employed.
      
    """
    print("Training Model...")
    model = Doc2Vec(vector_size = 50, min_count=2, epochs= 40)

    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    os.system("clear")
    queries = pd.read_json("./data/queries/queries.json")

    # Odd number of queries are training set
    learning_queries = queries.loc[queries.loc[:,"number"] % 2 != 0,:]

    similarity_matrix = pd.DataFrame(index=range(len(learning_queries)), columns=lst_index)

    # Calculate cosine similartiy between each doc, query pair
    print("Creating Training Set...")
    for i in range(learning_queries.shape[0]):

        print("For query {}...".format(2*i+1))

        query = process_query(pd.DataFrame(learning_queries.iloc[i,:]).T)

        query_vector = model.infer_vector(query[0])

        results = model.docvecs.most_similar([query_vector], topn=len(train_corpus))

        for (uid, score) in tqdm(results, total=len(results)):
            similarity_matrix.at[i, uid] = score
    os.system("clear")
    print("DONE")
    similarity_matrix.to_csv("./preprocessing/doc2vec_cosine_similarity_matrix.csv")



if __name__=="__main__": 
    main() 
