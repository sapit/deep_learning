
from utils import *
import csv
import pandas as pd


for i in range(1,11):
    print("Processing Batch " + str(i))
    #df = pd.read_csv("brexit/attempt3/brexit_labelled_tweets_batch_" + str(i) + ".csv", sep=';')
    df = pd.read_csv("brexit/attempt3/brexit_labelled_tweets_new_" + str(i) + ".csv", sep=';')
    h = list(filter(lambda s: eval_english(s[1])<0.7 , map(lambda a: [process_tweet(a[1]['text']), a[1]['emotion']], df.iterrows()))) 

    with open("brexit/attempt3/brexit_labelled_tweets_batch_" + str(i) + "_en.csv", 'w') as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["text", "emotion"])
        writer.writerows(h)

