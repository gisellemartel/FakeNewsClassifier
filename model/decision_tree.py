import numpy as np
import sklearn
import sklearn.tree
from sklearn.metrics import accuracy_score

import sys
sys.path.insert(1, '../')

import tools as tools

def decision_tree_train(X, y, d):
    print("Fitting data to DecisionTree, this may take a while...")
    model = sklearn.tree.DecisionTreeClassifier(max_depth=d,random_state=0)
    model.fit(X,y)
    return model

def decision_tree_predict(model, X):
    return model.predict(X)

def decision_tree_hyperparam_search(X, y, D):
    estimators = []
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for d in D:
        estimator = decision_tree_train(X,y,d)
        estimators.append(estimator)
        y_pred = decision_tree_predict(estimator,X)

        # calculate the accuracy
        a = accuracy_score(y,y_pred)*100
        print("{:.3f}% training accuracy for max_depth={:.3f}".format(a,d))

        if a > highest_accuracy:
            highest_accuracy = a
            best_estimator = estimator
            best_hyperparams = {"D": d}

    return estimators, highest_accuracy, best_estimator, best_hyperparams

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting Decision Tree Classifier ...\n")

    # set the hyperparams
    D = np.linspace(1,1000,20)

    # perform hyperparam search
    estimators, accuracy, best_estimator, hyperparams = decision_tree_hyperparam_search(X_train, y_train, D)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nDecision Tree overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("DecisionTree",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(accuracy, "DecisionTree", hyperparams)

    # use best estimator to make predictions
    y_pred = decision_tree_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "DecisionTree", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"DecisionTree")
    tools.plot_confusion_matrix(y_test,y_pred,"DecisionTree", True)
