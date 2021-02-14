import os
import json

all_results = {}

for txt in os.listdir("./eval"):


    if "doc2vec_" in txt:
        with open("./eval/{}".format(txt) , "r") as txtfile:
            results = txtfile.read().splitlines()

        report = {}
        for line in results:
            metric = line.split()[0]
            score = line.split()[2]

            report[metric] = score


        all_results[txt.split(".")[0]] = report["map"]

    
    
with open("./eval/report.json", "w+") as reportFile:
    json.dump(all_results, reportFile) 

        