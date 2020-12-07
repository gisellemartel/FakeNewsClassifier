import numpy as np
import sklearn
import sklearn.tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

np.random.seed(0)

import sys
sys.path.insert(1, '../')

import tools.tools as tools

def random_forest_train(X,y,d,n):
    estimator = RandomForestClassifier(max_depth=d,n_estimators=n,random_state=0)
    estimator.fit(X, y)
    return estimator

def random_forest_predict(model, X):
    return model.predict(X)

'''
    Trains each estimator with the given hyperparams, displays the training accuracy for each and returns all estimators
'''
def train_all_estimators(X, y, D, N):
    print("Fitting data to Random Forest Classifier, this may take a while...")
    estimators = []
    for d in D:
        for n in N:
            estimator = random_forest_train(X,y,d,n)
            estimators.append(estimator)
            y_pred = random_forest_predict(estimator,X)
            a = accuracy_score(y,y_pred)*100
            print("{:.3f}% training accuracy for max_depth={:.3f} n_estimators={:.3f}".format(a,d,n))

    return estimators

def perform_hyperparam_grid_search(X_train, y_train, param_grid):
    model = RandomForestClassifier(max_depth=1,n_estimators=1,random_state=0)
    grid_search = sklearn.model_selection.GridSearchCV(verbose=1, cv=5, param_grid=param_grid, estimator=model)
    grid_search.fit(X_train, y_train)
    return grid_search
    

def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./results/mock_results/")
    print("Testing Random Forest Classifier ...\n")

    # set the hyperparams
    D = np.linspace(2,14,7)
    N = np.linspace(2,20,10, dtype="int32")

    param_grid = {"max_depth":D, "n_estimators": N}

    # fetch all the estimators given the chosen hyperparameters
    estimators = train_all_estimators(X_train,y_train,D,N)

    # perform hyperparam search
    grid_search = perform_hyperparam_grid_search(X_train,y_train, param_grid)

    best_estimator = grid_search.best_estimator_
    hyperparams = grid_search.best_params_
    score = grid_search.best_score_*100

    tools.plot_feature_importances(X_train, best_estimator, "RandomForest", savefig=True)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nRandom Forest overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("RandomForest",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(score, "RandomForest", hyperparams)

    # use best estimator to make predictions
    y_pred = random_forest_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test.values, y_pred, "RandomForest", True)
    tools.display_prediction_scores(y_test.values,y_pred)
    tools.write_metrics_to_file(y_test.values,y_pred,"RandomForest")
    tools.plot_confusion_matrix(y_test.values,y_pred,"RandomForest", True)
