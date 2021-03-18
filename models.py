from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# label is the string of the column we want our model to predict
def split_data(df, target_label):
    column_names = ["Date", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]
    return train_test_split(df.drop(["Date", "daily_return", target_label], axis=1), df[target_label], test_size=0.25, shuffle=False,
                            random_state=0)


def run_random_forest(x_train, x_test, y_train, y_test, estimators):
    clf = RandomForestClassifier(random_state=0, n_estimators=estimators)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


def run_decision_tree(x_train, x_test, y_train, y_test, num_max_features):
    clf = DecisionTreeClassifier(random_state=0, max_features=16)
    clf.fit(x_train, y_train)
    return clf.score(x_test, y_test)


def main():
    merge = pd.read_pickle("data/merged.pkl")
    # I think we might need to shift this value down one? not sure though
    merge[["daily_return", "daily_gain"]] = merge[["daily_return", "daily_gain"]].shift(periods=1)
    # drop all rows with NaN values until we figure that out
    merge = merge.dropna()
    label = "daily_gain"
    merge[label] = merge[label].astype("bool")
    x_train, x_test, y_train, y_test = split_data(merge.dropna(), label)
    estimators = 100
    print(run_random_forest(x_train, x_test, y_train, y_test, estimators))
    num_max_features = x_train.shape[1]
    print(run_decision_tree(x_train, x_test, y_train, y_test, num_max_features))

    # Trial below

    column_names = ["Open", "Close", "Adj Close", "Volume", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]
    x_train, x_test, y_train, y_test = train_test_split(merge[column_names], merge["daily_gain"], test_size=0.25, shuffle=False,
                                                        random_state=0)
    estimators = 100
    clf = RandomForestClassifier(random_state=0, n_estimators=estimators)
    clf.fit(x_train, y_train)
    print(clf.score(x_test, y_test))




main()
