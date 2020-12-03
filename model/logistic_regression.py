import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, '../')

import tools as tools

def logistic_regression_train(X, y, c):
    print("Fitting data to LogisticRegression Classifier, this may take a while...")
    model = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=100, C=c)
    model.fit(X,y)
    return model

def logistic_regression_predict(model, X):
    return model.predict(X)

def logisitic_regression_hyperparam_search(X, y, C):
    estimators = []
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for c in C:
        estimator = logistic_regression_train(X,y,c)
        estimators.append(estimator)
        y_pred = logistic_regression_predict(estimator,X)

        # calculate the accuracy
        a = accuracy_score(y,y_pred)*100
        print("{:.1f}% training accuracy for C={:.3f}".format(a, c))

        if a > highest_accuracy:
            highest_accuracy = a
            best_estimator = estimator
            best_hyperparams = {"C": c}

    return estimators, highest_accuracy, best_estimator, best_hyperparams
    

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting Logistic Regression Classifier ...\n")

  # set the hyperparams
    C = np.logspace(-4,4,6)

    # perform hyperparam search
    estimators, accuracy, best_estimator, hyperparams = logisitic_regression_hyperparam_search(X_train, y_train, C)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nLogistic Regression overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("LogisticRegression",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(accuracy, "LogisticRegression", hyperparams)

    # use best estimator to make predictions
    y_pred = logistic_regression_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "LogisticRegression", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"LogisticRegression")
    tools.plot_confusion_matrix(y_test,y_pred,"LogisticRegression", True)


