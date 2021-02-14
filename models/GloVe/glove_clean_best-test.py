import os
import string
import json
import requests

import pandas as pd
import numpy as np
from scipy import spatial

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

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from glove import Corpus, Glove

stop_words = stopwords.words('english')
punctuations = list(string.punctuation) +["–"]

data = pd.read_csv("./data/text/raw_data.csv",encoding="utf-8") 
queries = pd.read_json("./data/queries/queries.json")
with open("./data/labels/labels.json","r") as f:
  labels = json.load(f)

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
    sentence = "".join([i if ord(i) < 128 else ' ' for i in sentence])
    for ch in punctuations:
      sentence = sentence.replace(ch," ")
    word_tokens = word_tokenize(sentence)
    filtered_sentence = [w.lower() for w in word_tokens if not w in stop_words and not any(ch.isdigit() for ch in w) and len(w) > 1]
    stems = [stemmer.stem(word) for word in filtered_sentence]
    return [stem for stem in stems if len(stem) > 1]

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

lst_index, lst_words = process_df(data)

#corpus = Corpus() 
#corpus.fit(lst_words, window=10)
#glove = Glove(no_components=50, random_state=42, learning_rate = 0.05)
#glove.fit(corpus.matrix, epochs=500, no_threads=16, verbose=False)
#glove.add_dictionary(corpus.dictionary)
#glove.save('glove.model')
glove = Glove.load("./models/GloVe/glove/pretrained_models/glove_500.model")

glove_doc_vectors = []
glove_doc_ids = []
for idx in range(len(lst_index)):
  doc_id = lst_index[idx]  
  word_vectors = []
  print(idx) if idx % 1000 == 0 else ""
  for word in lst_words[idx]:
    try:
      word_vectors.append(glove.word_vectors[glove.dictionary[word]])
    except:
      print("err: ",idx,word)
  try:
    glove_doc_vectors.append(np.array(word_vectors).mean(0))
    glove_doc_ids.append(doc_id)
  except:
    print(idx, word_vectors)

query_ids = []
query_texts = []
learning_queries = queries.loc[queries.loc[:,"number"] % 2 == 0,:]
for i in range(learning_queries.shape[0]): 
  query_texts.append([" ".join(x) for x in process_query(pd.DataFrame(learning_queries.iloc[i,:]).T)][0].split())
  query_ids.append(list(learning_queries["number"])[i])

glove_query_vectors = []
glove_query_ids = []
for idx in range(len(query_ids)):
  query_vectors = []
  query_id = query_ids[idx]
  for word in query_texts[idx]:
    query_vectors.append(glove.word_vectors[glove.dictionary[word]]) 
  try:
    glove_query_vectors.append(np.array(query_vectors).mean(0))
    glove_query_ids.append(query_id)
  except:
    print(idx, query_vectors)

norms = {}
for idq in range(1,len(glove_query_vectors)+1):
  index = 2*idq#+1
  norms[str(index)] = {}
  query = glove_query_vectors[idq-1]
  for idx in range(len(glove_doc_vectors)):
    doc_vector = glove_doc_vectors[idx]
    doc_id = glove_doc_ids[idx]
    try:
      norms[str(index)][doc_id] = cosine_similarity([doc_vector], [query]).flatten()[0]
    except: ""
  print(idq,"/",len(glove_query_vectors))
  norms[str(index)] = {k:v for k,v in sorted(norms[str(index)].items(), key=lambda item:item[1], reverse=True)}

pd.DataFrame(data=list(norms.values()),columns=norms["2"].keys(),index=list(norms.keys())).to_csv("./models/GloVe/glove/preprocessing/cosine_similarity_matrix_test.csv")
cosine_similarity_matrix = pd.read_csv("./models/GloVe/glove/preprocessing/cosine_similarity_matrix_test.csv", index_col=0)

results = ""

for idx, row in cosine_similarity_matrix.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + cosine_similarity_matrix.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(row[i])                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("./models/GloVe/glove/results/myresults_cos_500_test.txt", "w")
outfile.write(results)
outfile.close()

myqrels = ""

labels = {int(k):v for k,v in labels.items() if int(k)%2==0}      # we are interested in even topics, but labels start from 1

for key,value in sorted(labels.items()):
    s = str(key)         # topic ID
    s += " 0 "                  # a necessary field that is currently ignored by trec eval
    for i in value:
        r = s
        r += i[0]               # document ID
        r += " "
        r += str(i[1])               # similarity
        r += "\n"
        myqrels += r

outfile = open("./models/GloVe/glove/results/myqrels_cos_500_test.txt", "w")
outfile.write(myqrels)
outfile.close()