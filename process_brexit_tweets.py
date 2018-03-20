from semEvalCat import *
from keras.models import load_model
from utils import *
import csv

emotion_model = load_model("data/emotion-detection-weights-improvement-100-0.1098.hdf5")

df = pd.read_csv("brexit/total2.csv", sep=';', error_bad_lines=False)


tweets = df["text"]
batches = [tweets[i:i+len(tweets)//10] for i in range(0, len(tweets), len(tweets)//10) ]

for i in range(len(batches)):
    batch = batches[i]

    predictions = predictions_from_raw(batch, emotion_model, words)
    h = list(map(lambda a: [a[0], emotions[np.argmax(a[1])]] , zip(tweets, predictions)))

    with open("brexit/brexit_labelled_tweets_new" + str(i) + ".csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "emotion"])
        writer.writerows(h)