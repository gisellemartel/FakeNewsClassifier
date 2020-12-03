import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import sys
sys.path.insert(1, '../')

import tools as tools

def support_vector_machine_train(X,y,c,g,k):
    print("Fitting data to SVC, this may take a while...")
    svc = SVC(kernel=k, C=c, gamma=g, max_iter=100, random_state=0)
    svc.fit(X, y)
    return svc

def support_vector_machine_predict(svc, X):
    return svc.predict(X)

#  performs hyperparam search on SVC estimators, returns the best result
def svc_hyperparam_search(X, y, C, G, K):
    estimators = []
    highest_accuracy = 0
    best_estimator = None
    best_hyperparams = ()
    # find the best accuracy from the selection of hyperparams
    for c in C:
        for g in G:
            for k in K:
                svc = support_vector_machine_train(X,y,c,g,k)
                estimators.append(svc)
                y_pred = support_vector_machine_predict(svc,X)

                # calculate the accuracy
                a = accuracy_score(y,y_pred)*100
                print("{:.1f}% training accuracy for C={:.3f} gamma={:.3f} kernel={}".format(a,c,g,k))

                if a > highest_accuracy:
                    highest_accuracy = a
                    best_estimator = svc
                    best_hyperparams = {"c": c, "gamma": g}

    return estimators, highest_accuracy, best_estimator, best_hyperparams

def test_run(X_train, X_test, y_train, y_test):
    print("\nTesting SVM Classifier ...\n")

    # set the hyperparams
    C = np.logspace(-4,4,6)
    G = np.logspace(-4,4,6)
    K = ["rbf", "linear"]

    # perform hyperparam search
    estimators, accuracy, best_estimator, hyperparams = svc_hyperparam_search(X_train, y_train, C, G, K)

    # calculate the training and testing scores and plot the result
    trn_scores, test_scores = tools.calculate_estimator_scores([X_train, X_test, y_train, y_test], estimators)

    # calculate model overfitting
    overfitting = tools.determine_overfitting(trn_scores,test_scores)
    print("\nSVC overfitting: {:.3f}\n".format(overfitting))
    
    # plot the scores of each estimator
    tools.plot_estimator_scores("SVC",trn_scores,test_scores,True)
    
    # display details of best estimator
    tools.display_best_estimator(accuracy, "SVC", hyperparams)

    # use best estimator to make predictions
    y_pred = support_vector_machine_predict(best_estimator, X_test)

    tools.plot_predicted_labels(y_test, y_pred, "SVC", True)
    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"SVC")
    tools.plot_confusion_matrix(y_test,y_pred,"SVC", True)