import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import sys
sys.path.insert(1, '../')

import tools as tools

def support_vector_machine_train(X,y,gamma):
    print("Fitting data to SVC, this may take a while...")
    svc = SVC(kernel='rbf', gamma=gamma,random_state=0)
    svc.fit(X, y)
    return svc

def support_vector_machine_predict(svc, X):
    return svc.predict(X)

#  performs hyperparam search on SVC estimators, returns the best result
def svc_hyperparam_search(X, y, C, G):
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for c in C:
        for g in G:
            svc = support_vector_machine_train(X,y,g)
            y_pred = support_vector_machine_predict(svc,X)

            a = accuracy_score(y,y_pred)*100
            print("{:.1f}% training accuracy for C={:.1f} gamma={:.2f}".format(a, c, g))

            if a > highest_accuracy:
                highest_accuracy = a
                best_estimator = svc
                best_hyperparams = {"c": c, "gamma": g}
    
    return highest_accuracy, best_estimator, best_hyperparams
            


def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting SVM Classifier ...\n")

    C = np.logspace(1,4, num=4)
    G = np.logspace(-2,1, num=4)

    accuracy, estimator, hyperparams = svc_hyperparam_search(X_train, y_train, C, G)

    tools.display_best_estimator(accuracy, "SVC", hyperparams)

    y_pred = support_vector_machine_predict(estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "SVC", True)

    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"SVC")
    tools.plot_confusion_matrix(y_test,y_pred,"SVC", True)