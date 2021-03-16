# Import necessary libraries
import numpy as np
import matplotlib.pylab as plt # for plotting
import pandas as pd
import yahoofinance as yf
import nltk
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

### these only get run when loading for the first time  ###
def label_sentiment(df):
    neg_sent = []
    neu_sent = []
    pos_sent = []
    comp_sent = []

    print("starting sentiment analysis")
    for tweet in df['text']:
        # pull out and score contents
        score = SentimentIntensityAnalyzer().polarity_scores(tweet)
        neg_sent.append(score['neg'])
        neu_sent.append(score['neu'])
        pos_sent.append(score['pos'])
        comp_sent.append(score['compound'])

    df['Neg_Sent'] = neg_sent
    df['Neu_Sent'] = neu_sent
    df['Pos_Sent'] = pos_sent
    df['Comp_Sent'] = comp_sent

    df.to_pickle("data/sentiment_labelled.pkl")
    return df

def load_tweets(file_name):
    df = pd.read_csv(file_name)
    df.date = pd.to_datetime(df.date)
    df = df[df.date >= np.datetime64('2016-08-01')]
    df = df.sort_values(by=["date"], ascending=True)
    return df

def load_stocks():
    historical = yf.HistoricalPrices('SPY', '2016-08-01', '2021-01-20')
    dfs = historical.to_dfs()
    df_stocks = dfs['Historical Prices']
    df_stocks
    daily_return = df_stocks["Open"].pct_change(1)
    df_stocks["daily_return"] = daily_return
    df_stocks["daily_gain"] = df_stocks["daily_return"] > 0
    df_stocks.to_pickle("data/df_stocks.pkl")

    return df_stocks

def get_avgd_array(df):
    # convert to an np array and then organize tweets by day
    tweet_array = np.array(df)
    tweet_dict = {}
    avg_tweet_array = []

    for row in tweet_array:
        curr_date = row[7].date()
        if curr_date in tweet_dict:
            tweet_dict[curr_date].append(row)
        else:
            tweet_dict[curr_date] = [row]

    for day in tweet_dict.keys():

        avg_neg = 0
        avg_pos = 0
        avg_neu = 0
        avg_comp = 0
        avg_retweets = 0
        avg_favorites = 0
        for tweet in tweet_dict[day]:
            avg_neg += tweet[-4]
            avg_pos += tweet[-3]
            avg_neu += tweet[-2]
            avg_comp += tweet[-1]
            avg_favorites += tweet[5]
            avg_retweets += tweet[6]

        tweets_per_day = len(tweet_dict[day])
        avg_neg /= tweets_per_day
        avg_pos /= tweets_per_day
        avg_neu /= tweets_per_day
        avg_comp /= tweets_per_day
        avg_retweets /= tweets_per_day
        avg_favorites /= tweets_per_day

        elem = [avg_neg, avg_pos, avg_neu, avg_comp, avg_retweets, avg_favorites]
        avg_tweet_array.append(elem)

    return np.array(avg_tweet_array)


# https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
# x is target sequence to average
# w is number of indicies to average i.e. w = 3 is a 3 day rolling average
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_test_train_split(df):
    x_train, x_test, y_train, y_test = train_test_split(df.data, df.features,
                                                        test_size=0.25, shuffle=False,
                                                        random_state=0)
# def build_df(df, df_stocks):
#
#

def main():
    reprocess_data = False

    if reprocess_data:
        file_name = "data/trump-twitter.csv"
        df = load_tweets(file_name)
        df = label_sentiment(df)
        df_stocks = load_stocks()

    else:
        df = pd.read_pickle("data/sentiment_labelled.pkl")
        df_stocks = pd.read_pickle("data/df_stocks.pkl")

    avg_tweet_array = get_avgd_array(df)
    print(avg_tweet_array.shape)



if __name__ == '__main__':
    main()
