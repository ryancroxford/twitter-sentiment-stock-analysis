# Import necessary libraries
import numpy as np
import matplotlib.pylab as plt # for plotting
import pandas as pd
import sklearn
import yahoofinance as yf
import nltk


fileName = "./trump-twitter.csv"
df = pd.read_csv(file)
df.date = pd.to_datetime(df.date)
df = df[df.date >= np.datetime64('2016-08-01')]
df = df.sort_values(by=["date"], ascending=True)
df.shape

neg_sent = []
neu_sent = []
pos_sent = []
comp_sent = []

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

historical = yf.HistoricalPrices('SPY', '2016-08-01', '2021-01-20')
dfs = historical.to_dfs()
df_stocks = dfs['Historical Prices']
df_stocks
daily_return = df_stocks["Open"].pct_change(1)
df_stocks["daily_return"] = daily_return
df_stocks["daily_gain"] = df_stocks["daily_return"] > 0
