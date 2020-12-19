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

