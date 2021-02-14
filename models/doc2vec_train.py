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

arg_parser.add_argument("-s", "--size")
arg_parser.add_argument("-e", "--epochs")

args = arg_parser.parse_args()

def list2tagged(doc_list, id_list):
    tagged_documents = []
    
    for doc, uid in zip(doc_list, id_list):
        tagged_documents.append(TaggedDocument(doc, [uid]))

    return tagged_documents


class MonitorCallback(CallbackAny2Vec):
    def __init__(self, learning_queries, corpus_size, lst_index):
        self.epoch = 0
        self.learning_queries = learning_queries
        self.corpus_size = corpus_size
        self.lst_index = lst_index


    def on_epoch_end(self, model):
        self.epoch += 1
        print("Epoch : {}".format(self.epoch))

        if self.epoch%25 == 0:
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
        print("Loading Corpus")
        with open("./preprocessing/processed_data.pickle", "rb") as processedData:
            lst_index, lst_words = pickle.load(processedData)

    else:
        print("No data found...")
        exit()

    if path.exists("./preprocessing/training_queries.pickle"):
        print("Loading Train Data")
        with open("./preprocessing/training_queries.pickle", "rb") as processedData:
            training_queries = pickle.load(processedData)

    else:
        print("No data found...")
        exit()

    try:
        os.mkdir("./models/doc2vec")
    except:
        pass

    vector_size = int(args.size)
    epochs = int(args.epochs)

    os.system("clear")
    print("Converting Tagged Document...")
    train_corpus = list2tagged(lst_words, lst_index)
    corpus_size = len(train_corpus)
    os.system("clear")

    print("Training Model...")
    my_callback = MonitorCallback(training_queries, corpus_size, lst_index)
    print("Defining model...")
    model = Doc2Vec(vector_size = vector_size, min_count=0, epochs = epochs, callbacks= [my_callback])
    print("Building_vocab")
    model.build_vocab(train_corpus)
    print("Training model...")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    with open("./models/doc2vec/model_s{}_e{}.pickle".format(vector_size, epochs), "wb") as processedData:
        pickle.dump(model, processedData)


if __name__=="__main__": 
    main() 
