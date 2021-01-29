import pandas as pd
import json

import matplotlib.pyplot as plt


kmeans_results = pd.read_csv("./models/kmeans/kmeans_predictions.csv", index_col=0)

cosine_similarity = pd.read_csv("./preprocessing/cosine_similarity_matrix.csv", index_col=0)

labels_json = json.load(open("./data/labels/labels.json"))

results = ""

for idx, row in cosine_similarity.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + cosine_similarity.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(row[i])                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("myresults_sim.txt", "w")
outfile.write(results)
outfile.close()

myqrels = ""

labels_json = {int(k):v for k,v in labels_json.items() if int(k)%2==1}      # we are interested in even topics, but labels start from 1

for key,value in sorted(labels_json.items()):
    s = str((key-1)//2)         # topic ID
    s += " 0 "                  # a necessary field that is currently ignored by trec eval
    for i in value:
        r = s
        r += i[0]               # document ID
        r += " "
        r += str(i[1])               # similarity
        r += "\n"
        myqrels += r

outfile = open("myqrels_sim.txt", "w")
outfile.write(myqrels)
outfile.close()

