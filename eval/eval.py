import pandas as pd
import json

import matplotlib.pyplot as plt

def map(ground_truth, predictions):
    # TODO
    pass

def normalized_DCG(ground_truth, predictions):
    # TODO
    pass

def top10_precision(ground_truth, predictions):
    # TODO
    pass

kmeans_results = pd.read_csv("./models/kmeans/kmeans_predictions.csv", index_col=0)

labels_json = json.load(open("./data/labels/labels.json"))

results = ""

for idx, row in kmeans_results.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + kmeans_results.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(int(row[i]))                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("myresults.txt", "w")
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
        r += i[1]               # similarity
        r += "\n"
        myqrels += r

outfile = open("myqrels.txt", "w")
outfile.write(myqrels)
outfile.close()


result = 0


for idx, row in kmeans_results.iterrows():
    label_df = pd.DataFrame(labels_json[str(idx*2+1)], columns=["uid", "relevance"])
    
    label_df = label_df.merge(row.T, left_on="uid", right_on=row.T.index)
    
    label_df.columns = ["uid", "relevance", "prediction"]

    label_df["prediction"] = label_df["prediction"].astype(int)

    label_df["relevance"] = label_df["relevance"].astype(int)

    mask = label_df["relevance"] == label_df["prediction"]

    result += (sum(mask)*100 / len(label_df))


print(result/25)
