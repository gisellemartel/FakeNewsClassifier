import numpy as np
import sklearn
from sklearn.metrics import accuracy_score

np.random.seed(0)

import sys
sys.path.insert(1, '../')

import tools.tools as tools

def logistic_regression_train(X, y, c):
    model = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=100, C=c)
    model.fit(X,y)
    return model

def logistic_regression_predict(model, X):
    return model.predict(X)

'''
    Trains each estimator with the given hyperparams, displays the training accuracy for each and returns all estimators
'''
def train_all_estimators(X, y, C):
    print("Fitting data to LogisticRegression Classifier, this may take a while...")
    estimators = []
    for c in C:
        estimator = logistic_regression_train(X,y,c)
        estimators.append(estimator)
        y_pred = logistic_regression_predict(estimator,X)
        a = accuracy_score(y,y_pred)*100
        print("{:.3f}% training accuracy for C={:.6f}".format(a, c))
    return estimators

def perform_hyperparam_grid_search(X_train, y_train, param_grid):
    model = sklearn.linear_model.LogisticRegression(random_state=0, max_iter=100, C=0.001)
    grid_search = sklearn.model_selection.GridSearchCV(verbose=1, cv=5, param_grid=param_grid, estimator=model)
    grid_search.fit(X_train, y_train)
    return grid_search
    
def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./results/mock_results/")
    print("Testing Logistic Regression Classifier ...\n")

    # set the hyperparams
    C = np.logspace(-6,6,13)
    param_grid = {"C":C}

    # fetch all the estimators given the chosen hyperparameters
    estimators = train_all_estimators(X_train, y_train, C)

    # perform hyperparam search
    grid_search = perform_hyperparam_grid_search(X_train,y_train, param_grid)

    best_estimator = grid_search.best_estimator_
    hyperparams = grid_search.best_params_
    score = grid_search.best_score_*100

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nLogistic Regression overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("LogisticRegression",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(score, "LogisticRegression", hyperparams)

    # use best estimator to make predictions
    y_pred = logistic_regression_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test.values, y_pred, "LogisticRegression", True)
    tools.display_prediction_scores(y_test.values,y_pred)
    tools.write_metrics_to_file(y_test.values,y_pred,"LogisticRegression")
    tools.plot_confusion_matrix(y_test.values,y_pred,"LogisticRegression", True)
