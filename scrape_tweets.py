import tweepy
import pprint as pp
import time
import pickle
import json
import datetime
from keras.models import load_model
import numpy as np
from utils import *
pprint = pp.PrettyPrinter(indent=4).pprint
emotions = ["joy", "anger", "fear", "sadness"]

mashape_key = "QhjGzCLDY4mshX8m9kIubTv6KLb4p1TtzLJjsntXV9HI5WYc3q"
twitter_app_auth = {
    'consumer_key': '2NVO6JJl1Yig7XXuKMoa2PSkl',
    'consumer_secret': 'c9H6PAY9io3IlU3P3H44wihWCX80Ngo25d0RNGDWwGR9Zn6Utn',
    'access_token': '3131597903-eJp1XGXfXJj3iTnPYT0k50chm5p7RMCrZJJPPjG',
    'access_token_secret': 'Wo3iZaFhtDmdhHCAZNjYjFzjf14XwRb8ak93Qd0nVc1Dj',
  }

auth = tweepy.OAuthHandler(twitter_app_auth["consumer_key"], twitter_app_auth["consumer_secret"])
auth.set_access_token(twitter_app_auth["access_token"], twitter_app_auth["access_token_secret"])

api = tweepy.API(auth)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
#     print tweet.text
model_weights = "data/weights-categorical-1/emotion-detection-weights-improvement-21-0.2688.hdf5"
mymodel = load_model(model_weights)
words_file = "words_list.json"
with open(words_file, 'r') as read_file:
    words = json.load(read_file)
# MAX_TWEETS = 5000000000000000000000
MAX_TWEETS = 10
# MAX_TWEETS = 100000

tweets=set()
new_tweets = set()
since = '2000-02-16'
until = ''
while True:
    new_tweets = set()
    try:
        # tweets=set()
        for tweet in tweepy.Cursor(api.search, q='#flatearth',since='', until=until, rpp=100).items():
            if(tweet.text not in tweets):
                new_tweets.add(tweet.text)
                tweets.add(tweet.text)
            # tweets.add(tweet.text)
            # print(tweet.created_at)
            # last_created_at = tweet.created_at.strftime('%Y-%m-%d')
            until = tweet.created_at.strftime('%Y-%m-%d')
            # print(last_created_at)

            # print(tweet.text)
            # print(tweets)
            # print(new_tweets)
            # raise Exception
    except Exception as e:
        print (e)
        # print(e.strerror)
        if("status code = 429" in e.reason):
            time.sleep(61*15)

        if(len(new_tweets)==0):
            continue
        tweets_array = np.array(list(new_tweets))
        filtered_tweets_array = []
        new_tweets = None

        predictions = list(map(list,predictions_from_raw(tweets_array, mymodel, words)))

        for i in range(tweets_array.shape[0]):
            # if(max(predictions[i])>=0.7):
            if(max(predictions[i])>=0.0):
                emotion = emotions[predictions[i].index(max(predictions[i]))]
                filtered_tweets_array.append([tweets_array[i],emotion])
        # for i in range(tweets_array.shape[0]):
        if(len(filtered_tweets_array)==0):
            continue
        with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".pickle", "wb") as pickle_file:
            pickle.dump(filtered_tweets_array, pickle_file)

        with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".json", "w") as outfile:
            json.dump(list(filtered_tweets_array), outfile)
        time.sleep(61*15)
        # time.sleep(5)