import os
import subprocess

similarity_csv = []

for csv in os.listdir("./preprocessing/"):
    if "doc2vec" in csv and ".csv" in csv:
        similarity_csv.append(csv)


for csv in similarity_csv:
    os.system("python ./eval/eval.py ./preprocessing/{}".format(csv))

    proc = subprocess.Popen(["../trec_eval-9.0.7/trec_eval", "../cmpe493-term-project/myqrels_sim.txt", "../cmpe493-term-project/myresults_sim.txt"], stdout=subprocess.PIPE)
    out = proc.communicate()[0].decode("utf-8") 
    
    print(out)

    with open("./eval/{}.txt".format(csv[:-4]), "w+") as eval_result:
        eval_result.write(out)

