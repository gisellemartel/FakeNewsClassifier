import numpy as np
import sklearn
import sklearn.tree
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, '../')

import tools.tools as tools

def decision_tree_train(X, y, d):
    model = sklearn.tree.DecisionTreeClassifier(max_depth=d,random_state=0)
    model.fit(X,y)
    return model

def decision_tree_predict(model, X):
    return model.predict(X)

'''
    Trains each estimator with the given hyperparams, displays the training accuracy for each and returns all estimators
'''
def train_all_estimators(X, y, D):
    print("Fitting data to DecisionTree, this may take a while...")
    estimators = []
    for d in D:
        estimator = decision_tree_train(X,y,d)
        estimators.append(estimator)
        y_pred = decision_tree_predict(estimator,X)
        a = accuracy_score(y,y_pred)*100
        print("{:.3f}% training accuracy for max_depth={:.3f}".format(a,d))

    return estimators
            
def perform_hyperparam_grid_search(X_train, y_train, param_grid):
    model = sklearn.tree.DecisionTreeClassifier(max_depth=1,random_state=0)
    grid_search = sklearn.model_selection.GridSearchCV(verbose=1, cv=5, param_grid=param_grid, estimator=model)
    grid_search.fit(X_train, y_train)
    return grid_search

def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./test_results/")
    print("Testing Decision Tree Classifier ...\n")

    # set the hyperparams
    D = np.linspace(2,30,15)
    param_grid = {"max_depth":D}

    # fetch all the estimators given the chosen hyperparameters
    estimators = train_all_estimators(X_train, y_train, D)

    # perform hyperparam search
    grid_search = perform_hyperparam_grid_search(X_train,y_train, param_grid)

    best_estimator = grid_search.best_estimator_
    hyperparams = grid_search.best_params_
    score = grid_search.best_score_ * 100

    tools.plot_feature_importances(X_train, best_estimator, "DecisionTree", savefig=True)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nDecision Tree overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("DecisionTree",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(score, "DecisionTree", hyperparams)

    # use best estimator to make predictions
    y_pred = decision_tree_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "DecisionTree", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"DecisionTree")
    tools.plot_confusion_matrix(y_test,y_pred,"DecisionTree", True)
