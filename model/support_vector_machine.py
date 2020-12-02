import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

def support_vector_machine(X_train, X_test,  y_train, y_test):
    print("Fitting data to SVC, this may take a while...")
    svc = SVC(kernel='linear', random_state=1)
    svc.fit(X_train, np.ravel(y_train))
    y_pred = svc.predict(X_test)
    
    print("SVC Accuracy is:", metrics.accuracy_score(y_test, y_pred))
    print("Random Forest Mean Squared Error is: {}".format(np.sqrt(mean_squared_error(y_test, y_pred))))
    
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    labels =  np.array([[conf_matrix[0][0],conf_matrix[0][1]],[conf_matrix[1][0],conf_matrix[1][1]]])
    plt.title('Confusion matrix of the SVC classifier\n')
    sns.heatmap(conf_matrix, annot=labels, fmt = '', cmap = 'YlGnBu')
    
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])
    plt.savefig('./results/SVC/SVC_confusion_matrix.jpg')
    
    precision, recall, fscore, support = score(y_test,y_pred)
    
    with open('./results/SVC/SVC_metrics.txt', 'w') as file:
        file.write('False Precision : {:.4f}\n'.format(precision[0]))
        file.write('False Recall : {:.4f}\n'.format(recall[0]))
        file.write('False fscore : {:.4f}\n\n'.format(fscore[0]))
        file.write('True Precision : {:.4f}\n'.format(precision[1]))
        file.write('True Recall : {:.4f}\n'.format(recall[1]))
        file.write('True fscore : {:.4f}\n'.format(fscore[1]))
