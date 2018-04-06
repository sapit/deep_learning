from semEvalCat import *
from keras.models import load_model
from utils import *
import csv
import pandas as pd
emotion_model = load_model("data/emotion-detection-weights-improvement-100-0.1098.hdf5")

df = pd.read_csv("brexit/total2.csv", sep=';', error_bad_lines=False)


tweets = df["text"]
batches = [tweets[i:i+len(tweets)//10] for i in range(0, len(tweets), len(tweets)//10) ]

for i in range(len(batches)):
    print("Batch " + str(i))
    #if(i in [1,2]):
    #    continue
    batch = batches[i]

    predictions = predictions_from_raw(batch, emotion_model, words)
    #h = list(map(lambda a: [a[0], emotions[np.argmax(a[1])]] , filter(lambda a: max(a[1])>0.5 , zip(batch, predictions))))
    h = [(t,emotions[np.argmax(p)]) for t,p in zip(batch, predictions) if max(p)>0.5 ]

    with open("brexit/attempt3/brexit_labelled_tweets_new_" + str(i+1) + ".csv", "w") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["text", "emotion"])
        writer.writerows(h)
