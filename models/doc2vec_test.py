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

import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.callbacks import CallbackAny2Vec

from sklearn.metrics.pairwise import cosine_similarity

arg_parser = argparse.ArgumentParser()


arg_parser.add_argument("-p", "--path")

args = arg_parser.parse_args()

class MonitorCallback(CallbackAny2Vec):
    def __init__(self, learning_queries, corpus_size, lst_index):
        self.epoch = 0
        self.learning_queries = learning_queries
        self.corpus_size = corpus_size
        self.lst_index = lst_index


    def on_epoch_end(self, model):
        self.epoch += 1
        print("Epoch : {}".format(self.epoch))

        if self.epoch%1 == 0:
            similarity_matrix = pd.DataFrame(index=range(len(self.learning_queries)), columns=self.lst_index)

            # Calculate cosine similartiy between each doc, query pair
            print("Evaluating on the training set...")
            for i in range(len(self.learning_queries)):

                print("For query {}...".format(2*i+1))

                query_vector = model.infer_vector(self.learning_queries[i][0])

                results = model.docvecs.most_similar([query_vector], topn=self.corpus_size)

                for (uid, score) in tqdm(results, total=len(results)):
                    similarity_matrix.at[i, uid] = score
            os.system("clear")
            print("DONE")
            similarity_matrix.to_csv("./models/doc2vec/doc2vec_s{}_e{}_train.csv".format(model.vector_size, self.epoch))



def main():

    if path.exists("./preprocessing/processed_data.pickle"):
        with open("./preprocessing/processed_data.pickle", "rb") as processedData:
            lst_index, lst_words = pickle.load(processedData)

    
    if path.exists("./preprocessing/testing_queries.pickle"):
        print("Loading Test Data")
        with open("./preprocessing/testing_queries.pickle", "rb") as processedData:
            testing_queries = pickle.load(processedData)

    else:
        print("No data found...")
        exit()

    model_name = args.path

    epochs = model_name.split("_")[-1][1:]

    

    if path.exists("./models/doc2vec/{}.pickle".format(model_name)):
        print("Loading Model")
        with open("./models/doc2vec/{}.pickle".format(model_name), "rb") as processedData:
            model = pickle.load(processedData)

    else:
        print("No model found...")
        exit()

    #corpus_size = len(model.documents)

    similarity_matrix = pd.DataFrame(index=range(len(testing_queries)), columns=lst_index)

    # Calculate cosine similartiy between each doc, query pair
    print("Evaluating on the test set...")
    for i in range(len(testing_queries)):

        print("For query {}...".format(2*(i+1)))

        query_vector = model.infer_vector(testing_queries[i][0])

        results = model.docvecs.most_similar([query_vector], topn=len(lst_index))

        for (uid, score) in tqdm(results, total=len(results)):
            similarity_matrix.at[i, uid] = score
    os.system("clear")
    print("DONE")
    similarity_matrix.to_csv("./models/doc2vec/doc2vec_s{}_e{}_test.csv".format(model.vector_size, epochs))


if __name__=="__main__": 
    main() 

