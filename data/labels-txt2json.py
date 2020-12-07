import json


def line2list(line):
    # query_id:int, doc_id:string, relevance_class:int
    list_line = line.split()
    return [list_line[0], list_line[2], list_line[3]]


with open("labels.txt") as labelFile:
    lines = labelFile.readlines()

    lines = list(map(line2list, lines))

# key is query id, value is list of tuples (doc_id:string, relevance:int)
labels = {}

doc_list = []

for line in lines:
    if line[0] in labels.keys():
        labels[line[0]] += [ (line[1], line[2]) ]
    else:
        labels[line[0]] = [ (line[1], line[2]) ]

    doc_list.append(line[1])

with open("labels.json", "w+") as labeljson:
    json.dump(labels, labeljson)

doc_list = set(doc_list)

with open("doc_list.txt", "w+") as docFile:
    for doc in doc_list:
        docFile.write(doc+"\n")