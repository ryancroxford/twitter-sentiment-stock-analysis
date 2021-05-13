from sklearn import datasets, preprocessing, metrics
from sklearn.model_selection import train_test_split
import kerastuner as kt
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


# label is the string of the column we want our model to predict
def split_data(df, target_label):
    column_names = ["Date", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]

    x_train, dated_x_test, y_train, y_test = train_test_split(df.drop([target_label], axis=1), df[target_label], test_size=0.25, shuffle=False,
                                                              random_state=0)

    return *train_test_split(df.drop(["Date", "daily_return", target_label], axis=1), df[target_label], test_size=0.25, shuffle=False,
                             random_state=0), dated_x_test


def run_random_forest_classifer(x_train, x_test, y_train, y_test, estimators):
    clf = RandomForestClassifier(random_state=0, n_estimators=estimators)
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print(f"Random Forest Classifer score: {score}")
  # calc_score_on_high_prob(clf, x_test, y_test)
    y_pred = clf.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Random Forest Metrics:")
    calc_metrics(conf_matrix)
    plot_confusion_matrix(clf, x_test, y_test)
    plt.title("Confusion Matrix for Random Forest")
    plt.show()
    return y_pred

def run_decision_tree(x_train, x_test, y_train, y_test, num_max_features):
    clf = DecisionTreeClassifier(random_state=0, max_features=16)
    clf.fit(x_train, y_train)
    print(f"decision tree Classifer score: {clf.score(x_test, y_test)}")
    y_pred = clf.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Decision Tree Metrics:")
    calc_metrics(conf_matrix)
    plot_confusion_matrix(clf, x_test, y_test)
    plt.title("Confusion Matrix for Decision Tree")
    plt.show()
    # calc_score_on_high_prob(clf, x_test, y_test)
    return y_pred


def calc_score_on_high_prob(clf, x_test, y_test):
    # Quick experiment to see if just selecting indicies of the testing data that had a high probability
    # returns a high score
    probs = clf.predict_proba(x_test)
    # get a boolean array of the probability matrix where there is a value greater than 0.6
    boolOfHighProb = np.any(probs > 0.6, axis=1)
    x_high_prob, y_high_prob = x_test[boolOfHighProb], y_test[boolOfHighProb]
    print(
        f"Score of prediction when controlling for high probability {clf.score(x_high_prob, y_high_prob)}")
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


def calc_performance(x_test, y_pred, model_type, starting_balance):
    buyer = [starting_balance]
    agent = [starting_balance]
    random_walker = [starting_balance]
    dates = []
    stock = []
    x_test["Pred"] = y_pred.tolist()

    for index, row in x_test.iterrows():
        delta = row["Close"] - row["Open"]

        buyer_delta = delta
        agent_delta = delta if row["Pred"] else 0
        random_walker_delta = delta if round(random.random()) else 0

        buyer.append(buyer[-1] + buyer_delta)
        agent.append(agent[-1] + agent_delta)
        random_walker.append(random_walker[-1] + random_walker_delta)

        dates.append(row["Date"])
        stock.append((row["Open"] + row["Close"])/2)

    buyer = buyer[1:]
    agent = agent[1:]
    random_walker = random_walker[1:]


    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False)
    ax1.set_title(model_type + ' Performance')
    ax1.plot(dates, agent, '-b', label='Agent')
    ax1.plot(dates, buyer, '-g', label='Constant Buyer')
    ax1.plot(dates, random_walker, '-r', label='Random Walker')

    ax2.plot(dates, stock, '-k', label='Stock')

    leg1 = ax1.legend()
    leg2 = ax2.legend()
    plt.show()


def main():
    merge = pd.read_pickle("data/trump_merged.pkl")
    # I think we might need to shift this value down one? not sure though
    merge[["daily_return", "daily_gain"]] = merge[[
        "daily_return", "daily_gain"]].shift(periods=1)
    # drop all rows with NaN values until we figure that out
    merge = merge.dropna()
    label = "daily_gain"
    merge[label] = merge[label].astype("bool")

    x_train, x_test, y_train, y_test, dated_x_test = split_data(
        merge.dropna(), label)

    estimators = 100
    cols = merge[["Volume", "daily_gain", "three_day_comp"]]
    rf_y_pred = run_random_forest_classifer(
        x_train, x_test, y_train, y_test, estimators)

    calc_performance(dated_x_test, rf_y_pred,
                     model_type="Random Forest", starting_balance=500)

    num_max_features = x_train.shape[1]
    dt_y_pred = run_decision_tree(
        x_train, x_test, y_train, y_test, num_max_features)

    calc_performance(dated_x_test, dt_y_pred,
                     model_type="Decision Tree", starting_balance=500)


main()
