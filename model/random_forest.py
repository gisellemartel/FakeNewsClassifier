import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def Random_Forest(X_train, X_test,  y_train, y_test):
    clf = RandomForestClassifier(random_state = 1)

    depth = [50, 60, 70]
    print("This part may take a while...")
    for j in depth:
        print("\tEvaluating at depth value: {}".format(j))
        clf_GridSearch = GridSearchCV(estimator = clf,
                                      param_grid = {'n_estimators': [200], 'max_depth': [j]},
                                      cv = 5).fit(X_train, np.ravel(y_train))
    y_pred_clf = clf_GridSearch.predict(X_test)
    print("\t Random Forest Accuracy is:", metrics.accuracy_score(y_test, y_pred_clf))
    print("\t Random Forest Mean Squared Error is: {}\n".format(np.sqrt(mean_squared_error(y_test, y_pred_clf))))

    clf_GridSearch.best_params_
    
    conf_matrix = metrics.confusion_matrix(y_test, y_pred_clf)
    labels =  np.array([[conf_matrix[0][0],conf_matrix[0][1]],[conf_matrix[1][0],conf_matrix[1][1]]])
    plt.title('Confusion matrix of the RandomForest classifier\n')
    sns.heatmap(conf_matrix, annot=labels, fmt = '', cmap = 'YlGnBu')

    ax= plt.subplot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])
    plt.savefig('./results/RandomForest/RF_confusion_matrix.jpg')
    
    precision, recall, fscore, support = score(y_test,y_pred_clf)
    
    with open('./results/RandomForest/RF_metrics.txt', 'w') as file:
        file.write('False Precision : {:.2f}\n'.format(precision[0]))
        file.write('False Recall : {:.2f}\n'.format(recall[0]))
        file.write('False fscore : {:.2f}\n\n'.format(fscore[0]))
        file.write('True Precision : {:.2f}\n'.format(precision[1]))
        file.write('True Recall : {:.2f}\n'.format(recall[1]))
        file.write('True fscore : {:.2f}\n'.format(fscore[1]))
    plt.figure()
    
    feature_importances = pd.Series(clf_GridSearch.best_estimator_.feature_importances_, index=X_train.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Feature importances for top 10 words")
    plt.savefig('./results/RandomForest/RF_FeatureImportancesLargest10.jpg')
