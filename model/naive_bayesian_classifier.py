import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

import sys
sys.path.insert(1, '../')

import tools as tools

def naive_bayesian_train(X,y,a,f):
    print("Fitting data to NaiveBayes Classifier, this may take a while...")
    model = MultinomialNB(alpha=a, fit_prior=f)
    model.fit(X, y)
    return model

def naive_bayesian_predict(model, X):
    return model.predict(X)

def naive_bayes_hyperparam_search(X, y, A, F):
    estimators = []
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for a in A:
        for f in F:
            estimator = naive_bayesian_train(X,y,a,f)
            estimators.append(estimator)
            y_pred = naive_bayesian_predict(estimator,X)

            # calculate the accuracy
            acc = accuracy_score(y,y_pred)*100
            print("{:.1f}% training accuracy for alpha={:.3f} fit_prior={}".format(acc,a,f))

            if acc > highest_accuracy:
                highest_accuracy = acc
                best_estimator = estimator
                best_hyperparams = {"alpha": a, "fit_prior": f}

    return estimators, highest_accuracy, best_estimator, best_hyperparams

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting Naive Bayesian Classifier ...\n")
    
    # set the hyperparams
    A = np.linspace(0.05,1,12)
    F = [True, False]

    # perform hyperparam search
    estimators, accuracy, best_estimator, hyperparams = naive_bayes_hyperparam_search(X_train, y_train, A,F)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nNaive Bayes overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("NaiveBayes",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(accuracy, "NaiveBayes", hyperparams)

    # use best estimator to make predictions
    y_pred = naive_bayesian_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "NaiveBayes", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"NaiveBayes")
    tools.plot_confusion_matrix(y_test,y_pred,"NaiveBayes", True)

    tools.display_result(best_estimator, X_train)