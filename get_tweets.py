import tweepy as tw
import pandas as pd
CONSUMER_KEY = "0wB34Ld7N6PJPLsqIkWJVFU4w"
CONSUMER_SECRET = "cjsuvi8mn9II4wgH5pXGXML2AoOnMTKgA1srcd3PAKlXiCNrPB"


def get_user_tweets(user):

    auth = tw.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    api = tw.API(auth, wait_on_rate_limit=True)

    # a list to hold all of a users tweets
    user_tweets = []

    # can only get 200 tweets per request
    new_tweets = api.user_timeline(screen_name = user, count = 200)

    # save the most recent tweets
    user_tweets.extend(new_tweets)

    # save the id of the oldest tweet to continue from
    oldest = user_tweets[-1].id - 1

    # continue to grab tweets until done
    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(screen_name = user, count = 200, max_id = oldest)
        user_tweets.extend(new_tweets)
        oldest = user_tweets[-1].id - 1

    out_tweets = [[tweet.id_str, tweet.created_at, tweet.text] for tweet in user_tweets]