import tweepy as tw
import csv
import pandas as pd
import numpy as np

CONSUMER_KEY = "0wB34Ld7N6PJPLsqIkWJVFU4w"
CONSUMER_SECRET = "cjsuvi8mn9II4wgH5pXGXML2AoOnMTKgA1srcd3PAKlXiCNrPB"


def get_user_tweets(api, user):

    # a list to hold all of a users tweets
    user_tweets = []

    # can only get 200 tweets per request
    new_tweets = api.user_timeline(id=user, count=200)

    # save the most recent tweets
    user_tweets.extend(new_tweets)

    # save the id of the oldest tweet to continue from
    oldest = user_tweets[-1].id - 1

    # continue to grab tweets until done
    while len(new_tweets) > 0:
        new_tweets = api.user_timeline(id=user, count=200, max_id=oldest)
        user_tweets.extend(new_tweets)
        oldest = user_tweets[-1].id - 1

    out_tweets = [[user, tweet.created_at, tweet.text,
                   tweet.retweet_count, tweet.favorite_count] for tweet in user_tweets]

    with open(f'data/politician_tweets.csv', 'a', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_tweets)

    pass


def load_tweets(file_name):
    df = pd.read_csv(file_name)
    df.date = pd.to_datetime(df.date)
    df = df[df.date >= np.datetime64('2016-08-01')]
    df = df.sort_values(by=["date"], ascending=True)
    return df


def main():

    get_tweets = False

    if get_tweets:
        auth = tw.AppAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        api = tw.API(auth, wait_on_rate_limit=True)

        with open(f'data/politician_tweets.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["user", "date", "text", "retweets", "favorites"])

        with open(f'data/politician_handles.csv') as f:
            reader = csv.reader(f)
            start = False
            for row in reader:
                if row[2] == "Shelby, Richard":     # if api cuts off requests, change this name to the new start point
                    start = True
                if row[4] != "" and start:
                    print("Getting tweets for ", row[2], " ", row[0])
                    get_user_tweets(api, row[4])
                    print("Finished getting tweets for ", row[2], " ", row[0])


main()
