from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# label is the string of the column we want our model to predict
def split_data(df, target_label):
    column_names = ["Date", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]
    return train_test_split(df.drop(["Date", "daily_return", target_label], axis=1), df[target_label], test_size=0.25, shuffle=False,
                            random_state=0)


def run_random_forest_classifer(x_train, x_test, y_train, y_test, estimators):
    clf = RandomForestClassifier(random_state=0, n_estimators=estimators)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(f"Random Forest Classifer score: {score}")
  #  calc_score_on_high_prob(clf, x_test, y_test)
    pred = clf.predict(x_test)
    conf_matrix = confusion_matrix(pred, y_test)
    print("Random Forest Metrics:")
    calc_metrics(conf_matrix)


def run_decision_tree(x_train, x_test, y_train, y_test, num_max_features):
    clf = DecisionTreeClassifier(random_state=0, max_features=16)
    clf.fit(x_train, y_train)
    print(f"decision tree Classifer score: {clf.score(x_test, y_test)}")
    pred = clf.predict(x_test)
    conf_matrix = confusion_matrix(pred, y_test)
    print("Decision Tree Metrics:")
    calc_metrics(conf_matrix)
   # calc_score_on_high_prob(clf, x_test, y_test)


def calc_score_on_high_prob(clf, x_test, y_test):
    # Quick experiment to see if just selecting indicies of the testing data that had a high probability
    # returns a high score
    probs = clf.predict_proba(x_test)
    # get a boolean array of the probability matrix where there is a value greater than 0.6
    boolOfHighProb = np.any(probs > 0.6, axis=1)
    x_high_prob, y_high_prob = x_test[boolOfHighProb], y_test[boolOfHighProb]
    print(f"Score of prediction when controlling for high probability {clf.score(x_high_prob, y_high_prob)}")
    disp = plot_confusion_matrix(clf, x_high_prob, y_high_prob)
    plt.title("Confusion Matrix for p>0.6")
    plt.show()

def calc_metrics(conf_matrix):
    tp = conf_matrix[1][1]
    fp = conf_matrix[0][1]
    tn = conf_matrix[1][0]
    fn = conf_matrix[0][0]
    accuracy = (tp+tn)/(tp+fp+tn+fn)
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    print(f"Accuracy: {accuracy}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")


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
    run_random_forest_classifer(x_train, x_test, y_train, y_test, estimators)
    num_max_features = x_train.shape[1]
    run_decision_tree(x_train, x_test, y_train, y_test, num_max_features)

    # messing around with various things below

    column_names = ["Open", "Close", "Adj Close", "Volume", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]
    x_train, x_test, y_train, y_test = train_test_split(merge[column_names], merge["daily_gain"], test_size=0.2, shuffle=False,
                                                        random_state=0)
    estimators = 100
    clf = RandomForestClassifier(random_state=0, n_estimators=estimators)
    clf.fit(x_train, y_train)
    # get the actual predictions
    predictions = clf.predict(x_test)
    # conf_mat = confusion_matrix(y_test, predictions)

    # print(clf.score(x_test, y_test))
    # disp = plot_confusion_matrix(clf, x_test, y_test)
    # plt.title()
    # plt.show()

    # calc_score_on_high_prob(clf, x_test, y_test)

    # For the regressor, split the data with daily_return as the target
    x_train, x_test, y_train, y_test = train_test_split(merge[column_names], merge["daily_return"], test_size=0.2, shuffle=False,
                                                        random_state=0)



main()
