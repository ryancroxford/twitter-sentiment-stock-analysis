from hpsklearn import HyperoptEstimator, random_forest, any_classifier, any_preprocessing, decision_tree, svc, ada_boost
from hyperopt import tpe, hp
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(df, target_label):
    column_names = ["Date", "avg_neg", "avg_pos", "avg_neu", "avg_comp", "avg_retweets", "avg_favorites",
                    "three_day_neg", "three_day_pos", "three_day_neu", "three_day_comp", "three_day_retweets",
                    "three_day_favorites"]
    return train_test_split(df.drop(["Date", "daily_return", target_label], axis=1), df[target_label], test_size=0.25, shuffle=False,
                            random_state=0)

if __name__ == '__main__':
    merge_trump = pd.read_pickle("data/trump_merged.pkl")
    # I think we might need to shift this value down one? not sure though
    merge_trump[["daily_return", "daily_gain"]] = merge_trump[["daily_return", "daily_gain"]].shift(periods=1)
    # drop all rows with NaN values until we figure that out
    merge_trump = merge_trump.dropna()
    label = "daily_gain"
    merge_trump[label] = merge_trump[label].astype("bool")
    x_train, x_test, y_train, y_test = split_data(merge_trump.dropna(), label)

    # estim = HyperoptEstimator(classifier=random_forest('my_clf'))
    #                           # preprocessing=any_preprocessing('my_pre'),
    #                           # algo=tpe.suggest,
    #                           # max_evals=100,
    #                           # trial_timeout=120)
    clf_T = hp.pchoice('classifier_trump',
                     [(0.25, random_forest('classifier_trump.random_forest')),
                      (0.25, decision_tree('classifier_trump.random_forest')),
                      (0.25, svc('classifer_trump.svc')),
                      (0.25, ada_boost('classifer_trump.knn'))])

    estimT = HyperoptEstimator(classifier=clf_T)
    estimT.fit(x_train, y_train)
    print(f"Score for Trumps on hyperopt: {estimT.score(x_test, y_test)}")
    print(estimT.best_model())

    merge_politicians = pd.read_pickle("data/politician_merged.pkl")

    merge_politicians[["daily_return", "daily_gain"]] = merge_politicians[["daily_return", "daily_gain"]].shift(periods=1)
    # drop all rows with NaN values until we figure that out
    merge_politicians = merge_politicians.dropna()
    label = "daily_gain"
    merge_politicians[label] = merge_politicians[label].astype("bool")
    x_train, x_test, y_train, y_test = split_data(merge_politicians.dropna(), label)

    # estim = HyperoptEstimator(classifier=random_forest('my_clf'))
    #                           # preprocessing=any_preprocessing('my_pre'),
    #                           # algo=tpe.suggest,
    #                           # max_evals=100,
    #                           # trial_timeout=120)
    clf_P = hp.pchoice('classifier_politicians',
                     [(0.25, random_forest('classifier_politicians.random_forest')),
                      (0.25, decision_tree('classifier_politicians.random_forest')),
                      (0.25, svc('classifier_politicians.svc')),
                      (0.25, ada_boost('classifier_politicians.knn'))])

    estimP = HyperoptEstimator(classifier=clf_P)
    estimP.fit(x_train, y_train)
    print(f"Score for Politicians on hyperopt: {estimP.score(x_test, y_test)}")
    print(estimP.best_model())
