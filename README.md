# twitter-sentinment-stock-analysis

## Dependencies

1. numpy
2. matplotlib
3. pandas
4. sklearn
5. yahoofinance
6. nltk
7. tensorflow

## Data

[Trump Tweets](https://www.thetrumparchive.com/)
[Congressional Handles](https://triagecancer.org/congressional-social-media)
[Twitter API](https://developer.twitter.com/en/docs/twitter-api)
[Stock Data](https://pypi.org/project/yfinance/)

**data/** contains the data at several stages of processing, to avoid unnecessary recomputation. It contains financial data for the S&P 500, as well as the daily twitter activities of several politicians from 2016-2021. It also contains processed and merged versions of those datasets, where the tweets are joined with the stock performance by date, and additional moving averages and metrics are added. The tweets are also given sentiment score labels with their own additional metrics.




## Run
1.  `data_clean.py` handles the processing for the data. Based on whether or not processing has already occured, it will either scrape the data from the input csvs and yahoofinance and perform the joins and sentiment analysis, or it will load in the pickled data from the previous run. To load in pre-processed data, set `reprocess_data = False` in `main()`. It then separates out our chosen politicians for use and sets up the final dataframes.
2.  `models.py` and `sentiment_svm.ipynb` contain our models. The models run in turn, some of them performing automated hyperparameter tuning. They then report their results and metrics for analysis.
