import pandas as pd

okapi_scores = pd.read_csv("okapi_matrix.csv", index_col=0)
okapi_scores_L = pd.read_csv("okapi_matrix_L.csv", index_col=0)
okapi_scores_Plus = pd.read_csv("okapi_matrix_Plus.csv", index_col=0)

results = ""
for idx, row in okapi_scores.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + okapi_scores.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(row[i])                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("results_bm25.txt", "w")
outfile.write(results)
outfile.close()

results = ""
for idx, row in okapi_scores_L.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + okapi_scores_L.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(row[i])                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("results_bm25_L.txt", "w")
outfile.write(results)
outfile.close()

results = ""
for idx, row in okapi_scores_Plus.iterrows():
    s = str(idx)
    s+= " Q0 "                                  # a necessary field that is currently ignored by trec eval
    for i in range(0,len(row)):
        r = s + okapi_scores_Plus.columns[i]       # document ID
        r += " 0 "                              # rank, currently ignored by trec eval
        r += str(row[i])                   # score, similarity of the query and the document
        r += " STANDARD\n"
        results += r

outfile = open("results_bm25_Plus.txt", "w")
outfile.write(results)
outfile.close()


