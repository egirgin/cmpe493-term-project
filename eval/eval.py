import pandas as pd
import json

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

for idx, row in kmeans_results.iterrows():
    label_df = pd.DataFrame(labels_json[str(idx*2+1)], columns=["uid", "relevance"])

    label_df = label_df.merge(row.T, left_on="uid", right_on=row.T.index)

    label_df.columns = ["uid", "relevance", "prediction"]

    mask = label_df["relevance"] == label_df["prediction"]

    print(idx*2+1)
    print(sum(mask)*100 / len(label_df))


    




#print(label_df.head())

