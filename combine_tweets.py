import csv
import pandas as pd

emotions = ["joy", "anger", "fear", "sadness"]
outputs = {e:[] for e in emotions}
for i in range(len(dfs)):
    df= dfs[i]
    for e in emotions:
        f = df['emotion'] == e
        t = list(zip(df[f]['text'], [e]*len(df[f])))
        outputs[e].extend(t)


for e in outputs.keys():
    h = outputs[e]
    with open("brexit/attempt3/brexit_labelled_tweets_" + e + ".csv", "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["text", "emotion"])
        writer.writerows(h)

