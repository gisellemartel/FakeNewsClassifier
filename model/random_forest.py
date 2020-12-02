import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
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

def random_forest_train(X,y,D):
    models=[]
    print("This part may take a while...")
    for d in D: 
        models.append(random_forest_train_at_depth(X,y,d))
    return models

def random_forest_predict(models, X):
    predictions=[]
    for m in models:
        predictions.append(m.predict(X))
    return predictions

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting Random Forest Classifier ...\n")

    model = random_forest_train(X_train, y_train, [50, 60, 70])
    Y_pred = random_forest_predict(model, X_test)

    # tools.plot_predicted_labels(y_test, y_pred, "NaiveBayes", True)

    # # TODO: should print result for all depths
    # for y_pred in Y_pred:
        
    #     tools.display_prediction_scores(y_test,y_pred)
    #     tools.write_metrics_to_file(y_test,y_pred,"RandomForest")
    #     tools.plot_confusion_matrix(y_test,y_pred,"RandomForest", True)
    #     tools.plot_feature_importances(X_train, model[0].best_estimator_, "RandomForest", True)