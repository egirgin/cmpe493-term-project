import pandas as pd

metadata = pd.read_csv("./data/text/metadata.csv", low_memory=False)

raw_data = metadata[["cord_uid", "title", "abstract"]]

print("Original shape: " + str(raw_data.shape))

with open("./data/doc_list.txt" ,"r") as docFile:
    doc_list = docFile.readlines()

doc_list = list(map(lambda x: x[:-1], doc_list))

raw_data = raw_data[raw_data["cord_uid"].isin(doc_list)]


print("New shape: " + str(raw_data.shape))

raw_data.to_csv("./data/text/raw_data.csv")
