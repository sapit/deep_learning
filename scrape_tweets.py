import tweepy
import pprint as pp

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
tweets=[]
for tweet in tweepy.Cursor(api.search, q='#trump', rpp=100).items(MAX_TWEETS):
    # print "start\n %s \nend \n"%(tweet.text)
    tweets.append(tweet.text)
pprint(tweets)