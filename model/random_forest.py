import numpy as np
import sklearn
import sklearn.tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import sys
sys.path.insert(1, '../')

import tools as tools

def random_forest_train_at_depth(X,y,depth):
    estimator = RandomForestClassifier(random_state=0)
    print("Performing GridSearch with 200 estimators at depth value: {}".format(depth))
    param_grid = {'n_estimators': [200], 'max_depth': [depth]}
    return GridSearchCV(estimator=estimator,param_grid=param_grid,cv=5).fit(X,y)

def random_forest_train(X,y,d,n):
    print("Fitting data to Random Forest Classifier, this may take a while...")
    estimator = RandomForestClassifier(max_depth=d,n_estimators=n,random_state=0)
    estimator.fit(X, y)
    return estimator

def random_forest_predict(model, X):
    return model.predict(X)

def random_forest_hyperparam_search(X, y, D,N):
    estimators = []
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for d in D:
        for n in N:
            estimator = random_forest_train(X,y,d,n)
            estimators.append(estimator)
            y_pred = random_forest_predict(estimator,X)

            # calculate the accuracy
            a = accuracy_score(y,y_pred)*100
            print("{:.1f}% training accuracy for max_depth={:.3f} n_estimators={:.3f}".format(a,d,n))

            if a > highest_accuracy:
                highest_accuracy = a
                best_estimator = estimator
                best_hyperparams = {"D": d}

    return estimators, highest_accuracy, best_estimator, best_hyperparams

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting Random Forest Classifier ...\n")

    # set the hyperparams
    D = np.linspace(1,1000,20)
    N = np.linspace(100,2000,100)

    # perform hyperparam search
    estimators, accuracy, best_estimator, hyperparams = random_forest_hyperparam_search(X_train, y_train, D, N)

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
    y_pred = random_forest_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "DecisionTree", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"DecisionTree")
    tools.plot_confusion_matrix(y_test,y_pred,"DecisionTree", True)

