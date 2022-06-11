import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import chi2
import plott
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import time
from scipy.stats import mannwhitneyu
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

n_best_features = 5
def model_run(x_data,y_teacher):
    tic = time.time()
    ## Data transforming
    feat_mat = x_data.iloc[:, 1:]
    # feat_mat = normalize(feat_mat, axis=0, norm='l2')# Normlize data
    y = np.round_(np.array(y_teacher).astype(int),decimals =-2)
    # apply SelectKBest class to extract top n_best_features best features
    bestfeatures = SelectKBest(score_func=chi2, k=n_best_features)
    fit = bestfeatures.fit(feat_mat, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(feat_mat.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    print(featureScores.nlargest(n_best_features, 'Score'))  # print n_best_features best features
    plott.barr(featureScores.nlargest(n_best_features, 'Score').values[:, 1],
               featureScores.nlargest(n_best_features, 'Score').values[:, 0], 'Features Score',
               'Feature Score vs Features \n chi^2', 'chi12')

    # best_feat_names = featureScores.nlargest(n_best_features, 'Score').values[:, 0]  # Get best features
    # for feat in best_feat_names:  # Run over best features
    #     which_col = x_data.columns.str.match(feat)
    #     ids = np.where(x_data.values[:, which_col] > 0)[0]
    #     for col in range(np.shape(data_demographics.values[ids, 1:])[1]):  # Run over demographic
    #         col_demo = data_demographics.values[ids, col]
    #         col_demo_all = data_demographics.values[:, col]
    #         unique_val = np.unique(col_demo)
    #         for unique_v in unique_val:  # Run over unique values of column
    #             percent_unique_val = np.sum(unique_v == col_demo) / np.shape(col_demo)
    #             percent_unique_val_all = np.sum(unique_v == col_demo_all) / np.shape(col_demo_all)
    #             if mannwhitneyu(percent_unique_val,
    #                             percent_unique_val_all).pvalue < p_val:  # Is there significant difference
    #                 print("Significant demo found : {0}".format(data_demographics.columns[col]))

    # get a list of models to evaluate
    def get_models():
        models = dict()
        models['lr'] = LogisticRegression()
        models['knn'] = KNeighborsClassifier()
        models['cart'] = DecisionTreeClassifier()
        models['svm'] = SVC(decision_function_shape="ovo")
        models['bayes'] = GaussianNB()
        models['RF'] = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
        # models['NeuralNetworks'] = MLPClassifier(solver='sgd', alpha=1e-6, hidden_layer_sizes=(np.shape(feat_mat,1)*5, 10), random_state=1)
        return models

    # evaluate a given model using cross-validation
    def evaluate_model(model, X, y):
        cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=50, random_state=1)
        scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        return scores

    # get the models to evaluate
    models = get_models()
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, feat_mat, y)
        results.append(scores)
        names.append(name)
        print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
    # plot model performance for comparison
    plt.figure()
    plt.boxplot(results, labels=names, showmeans=True)
    plt.savefig('all_models.png')
    plt.show(block=False)
    print("Run Time Total {0}".format(time.time() - tic))
