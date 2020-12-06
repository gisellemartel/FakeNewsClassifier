import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

np.random.seed(0)

import sys
sys.path.insert(1, '../')

import tools.tools as tools

def naive_bayesian_train(X,y,a,f):
    model = MultinomialNB(alpha=a, fit_prior=f)
    model.fit(X, y)
    return model

def naive_bayesian_predict(model, X):
    return model.predict(X)

'''
    Trains each estimator with the given hyperparams, displays the training accuracy for each and returns all estimators
'''
def train_all_estimators(X, y, A, F):
    print("Fitting data to NaiveBayes Classifier, this may take a while...")
    estimators = []
    for f in F:
        for a in A:
            estimator = naive_bayesian_train(X,y,a,f)
            estimators.append(estimator)
            y_pred = naive_bayesian_predict(estimator,X)
            acc = accuracy_score(y,y_pred)*100
            print("{:.3f}% training accuracy for alpha={:.4f} fit_prior={}".format(acc,a,f))

    return estimators

def perform_hyperparam_grid_search(X_train, y_train, param_grid):
    model = MultinomialNB(alpha=1.0, fit_prior=True)
    grid_search = sklearn.model_selection.GridSearchCV(verbose=1, cv=5, param_grid=param_grid, estimator=model)
    grid_search.fit(X_train, y_train)
    return grid_search

def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./test_results/")
    print("Testing Naive Bayesian Classifier ...\n")
    
    # set the hyperparams
    A = np.logspace(-4,4,9)
    F = [True, False]
    param_grid = {"alpha":A, "fit_prior":F}

    # fetch all the estimators given the chosen hyperparameters
    estimators = train_all_estimators(X_train, y_train, A , F)

    # perform hyperparam search
    grid_search = perform_hyperparam_grid_search(X_train,y_train, param_grid)

    best_estimator = grid_search.best_estimator_
    hyperparams = grid_search.best_params_
    score = grid_search.best_score_*100

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nNaive Bayes overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("NaiveBayes",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(score, "NaiveBayes", hyperparams)

    # use best estimator to make predictions
    y_pred = naive_bayesian_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "NaiveBayes", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"NaiveBayes")
    tools.plot_confusion_matrix(y_test,y_pred,"NaiveBayes", True)

    tools.display_result(best_estimator, X_train)
