import tweepy
import pprint as pp
import time
import pickle
import json
import datetime
pprint = pp.PrettyPrinter(indent=4).pprint

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

# MAX_TWEETS = 5000000000000000000000
MAX_TWEETS = 10
# MAX_TWEETS = 100000
tweets=set()
since = '2000-02-16'
until = ''
while True:
    try:
        # tweets=set()
        for tweet in tweepy.Cursor(api.search, q='#flatearth',since='', until='', rpp=100).items():
            tweets.add(tweet.text)
            # print(tweet.created_at)
            last_created_at = tweet.created_at.strftime('%Y-%m-%d')
            print(last_created_at)
            # print(tweet.text)
            # raise Exception
    except Exception as e:
        print (e)
        pickle.dump(tweets, open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".pickle", "wb"))
        with open("tweets_dataset/tweets_" + str(int(time.time()*10)) + ".json", "w") as outfile:
            json.dump(list(tweets), outfile)
        time.sleep(60*15)
        # time.sleep(5)