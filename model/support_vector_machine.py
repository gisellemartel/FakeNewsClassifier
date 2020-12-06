import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import sys
sys.path.insert(1, '../')

import tools.tools as tools

def support_vector_machine_train(X,y,c,g,k):
    svc = SVC(kernel=k, C=c, gamma=g, max_iter=200, random_state=0)
    svc.fit(X, y)
    return svc

def support_vector_machine_predict(svc, X):
    return svc.predict(X)

'''
    Trains each estimator with the given hyperparams, displays the training accuracy for each and returns all estimators
'''
def train_all_estimators(X, y, C, G, K):
    print("Fitting data to SVC, this may take a while...")
    estimators = []
    for k in K:
        for c in C:
            for g in G:
                svc = support_vector_machine_train(X,y,c,g,k)
                estimators.append(svc)
                y_pred = support_vector_machine_predict(svc,X)
                a = accuracy_score(y,y_pred)*100
                print("{:.3f}% training accuracy for C={:.3f} gamma={:.3f} kernel={}".format(a,c,g,k))

    return estimators

def perform_hyperparam_grid_search(X_train, y_train, param_grid):
    model = SVC(max_iter=100, random_state=0)
    grid_search = sklearn.model_selection.GridSearchCV(verbose=1, cv=5, param_grid=param_grid, estimator=model)
    grid_search.fit(X_train, y_train)
    return grid_search
    
def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./results/mock_results/")
    print("Testing SVM Classifier ...\n")

    # set the hyperparams
    C = np.logspace(-2,2,5)
    G = np.logspace(-2,2,5)
    K = ["rbf", "linear"]

    param_grid = {"C":C, "gamma": G, "kernel": K}

    # fetch all the estimators given the chosen hyperparameters
    estimators = train_all_estimators(X_train, y_train, C, G, K)

    # perform hyperparam search
    grid_search = perform_hyperparam_grid_search(X_train,y_train, param_grid)

    best_estimator = grid_search.best_estimator_
    hyperparams = grid_search.best_params_
    score = grid_search.best_score_*100

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nSVC overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("SVC",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(score, "SVC", hyperparams)

    # use best estimator to make predictions
    y_pred = support_vector_machine_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "SVC", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"SVC")
    tools.plot_confusion_matrix(y_test,y_pred,"SVC", True)
