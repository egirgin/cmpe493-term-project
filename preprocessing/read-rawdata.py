import pandas as pd

metadata = pd.read_csv("../data/raw_data.csv")

data = []

for index, row in metadata.iterrows():

    paper = {
        "cord_uid" : row["cord_uid"],
        "title" : row["title"],
        "abstract" : row["abstract"]
    }
    data.append(paper)

print(len(data))