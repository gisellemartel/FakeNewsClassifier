import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_fscore_support as score
warnings.filterwarnings("ignore")

def Naive_Bayesian(X_train, X_test,  y_train, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, np.ravel(y_train))
    y_pred = clf.predict(X_test)
    
    print("Model accuracy is: {:#.3g}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print("Model mean Squared Error is: {:#.3g}%".format(np.sqrt(mean_squared_error(y_test, y_pred) *100)))
    
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)
    labels =  np.array([[conf_matrix[0][0],conf_matrix[0][1]],[conf_matrix[1][0],conf_matrix[1][1]]])
    plt.title('Confusion matrix of the naive bayes classifier\n')
    sns.heatmap(conf_matrix, annot=labels, fmt = '', cmap = 'YlGnBu')
    
    ax= plt.subplot()
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('Actual labels')
    ax.xaxis.set_ticklabels(['True', 'False'])
    ax.yaxis.set_ticklabels(['True', 'False'])
    plt.savefig('./results/NaiveBayes/NB_confusion_matrix.jpg')
    
    precision, recall, fscore, support = score(y_test,y_pred)
    
    with open('./results/NaiveBayes/NB_metrics.txt', 'w') as file:
        file.write('False Precision : {:.2f}\n'.format(precision[0]))
        file.write('False Recall : {:.2f}\n'.format(recall[0]))
        file.write('False fscore : {:.2f}\n\n'.format(fscore[0]))
        file.write('True Precision : {:.2f}\n'.format(precision[1]))
        file.write('True Recall : {:.2f}\n'.format(recall[1]))
        file.write('True fscore : {:.2f}\n'.format(fscore[1]))
    
    #Storing the number of times each token occurs in a True article
    true_token_count = clf.feature_count_[0, :]

    #Storing the number of times each token appears in a Fake article
    fake_token_count = clf.feature_count_[1, :]

    #create a dataframe out of the new data
    tokens = pd.DataFrame({'token':X_train.columns, 'true':true_token_count, 'fake':fake_token_count}).set_index('token')
    tokens['true'] = tokens.true + 1 #avoid division by 0 when doing frequency calculations
    tokens['fake'] = tokens.fake + 1
    tokens['true'] = tokens.true / clf.class_count_[0]
    tokens['fake'] = tokens.fake / clf.class_count_[1]

    tokens['fake/true ratio'] = tokens.fake / tokens.true
    tokens.sort_values('fake/true ratio', ascending=False).head(10)
