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
emotions = ["joy",
            "sadness",
            "anger",
            "fear"]

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

def store_tweets(new_tweets):
    pass
    tweets_array = np.array(list(new_tweets))
    filtered_tweets_array = []
    new_tweets = None

    predictions = list(map(list, predictions_from_raw(tweets_array, mymodel, words)))

    for i in range(tweets_array.shape[0]):
        # if(max(predictions[i])>=0.7):
        print(max(predictions[i]))
        if(max(predictions[i])>=0.5):
            # print("HUI2")
            print(tweets_array[i])
            # print(max(predictions[i]))
            emotion = emotions[predictions[i].index(max(predictions[i]))]
            filtered_tweets_array.append([tweets_array[i],emotion])
        # if(max(predictions[i])>=0.5):
        #     print("HUI2")

    # for i in range(tweets_array.shape[0]):
    if(len(filtered_tweets_array)==0):
        print("empty filtered_tweets_array")
        return
    # with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".pickle", "wb") as pickle_file:
    #     pickle.dump(filtered_tweets_array, pickle_file)

    with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".json", "w") as outfile:
        print("DUMPING")
        json.dump(list(filtered_tweets_array), outfile)
    # time.sleep(60*5)
    # time.sleep(5)

# public_tweets = api.home_timeline()
# for tweet in public_tweets:
model_weights = "data/weights-categorical-2/emotion-detection-weights-improvement-100-0.1098.hdf5"
mymodel = load_model(model_weights)
words_file = "words_list.json"
with open(words_file, 'r') as read_file:
    words = np.array(json.load(read_file))
# MAX_TWEETS = 5000000000000000000000
MAX_TWEETS = 10
# MAX_TWEETS = 100000

tweets=set()
new_tweets = set()
since = '2000-02-16'
until = datetime.date.today()
cursor = tweepy.Cursor(api.search, q='#flatearth', rpp=100, tweet_mode='extended').items()
iteration=0
new_tweets = set()
while True:
    iteration+=1
    # new_tweets = set()
    try:
        # tweets=set()
        # for tweet in tweepy.Cursor(api.search, q='#flatearth',since='', until=until.strftime('%Y-%m-%d'), rpp=100, tweet_mode='extended').items():
            tweet=cursor.next()
            if(tweet.lang == "en" and tweet.full_text not in tweets):
                print("Adding tweet")
                new_tweets.add(tweet.full_text)
                tweets.add(tweet.full_text)
            # tweets.add(tweet.full_text)
            # print(tweet.created_at)
            # last_created_at = tweet.created_at.strftime('%Y-%m-%d')
            until = tweet.created_at
            # print(last_created_at)
            # print(tweets)
            # print(new_tweets)
            # raise tweepy.TweepError("random")
    # except Exception as e:
    except tweepy.TweepError as e:
        print (e)
        print("HUI")
        # print(e.strerror)
        # if("status code = 429" in e.reason):
        print(len(new_tweets))
        if(len(new_tweets) > 0):
            store_tweets(new_tweets)
        time.sleep(61*5)
        new_tweets = set()
            # continue

        # if(iteration >=1):
        #     # print("HUI")
        #     iteration=0
        #     until = until - datetime.timedelta(1)


        # tweets_array = np.array(list(new_tweets))
        # filtered_tweets_array = []
        # new_tweets = None

        # predictions = list(map(list, predictions_from_raw(tweets_array, mymodel, words)))

        # for i in range(tweets_array.shape[0]):
        #     # if(max(predictions[i])>=0.7):
        #     print(max(predictions[i]))
        #     if(max(predictions[i])>=0.5):
        #         # print("HUI2")
        #         print(tweets_array[i])
        #         # print(max(predictions[i]))
        #         emotion = emotions[predictions[i].index(max(predictions[i]))]
        #         filtered_tweets_array.append([tweets_array[i],emotion])
        #     # if(max(predictions[i])>=0.5):
        #     #     print("HUI2")

        # # for i in range(tweets_array.shape[0]):
        # if(len(filtered_tweets_array)==0):
        #     continue
        # # with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".pickle", "wb") as pickle_file:
        # #     pickle.dump(filtered_tweets_array, pickle_file)

        # with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".json", "w") as outfile:
        #     print("DUMPING")
        #     json.dump(list(filtered_tweets_array), outfile)
        # # time.sleep(60*5)
        # # time.sleep(5)
    except Exception as e:
        print(e)
        pass

