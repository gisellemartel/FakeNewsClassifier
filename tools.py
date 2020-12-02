import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import precision_recall_fscore_support as score

RESULTS_DIR = "./results/"

def display_prediction_scores(test,prediction):
    accuracy = sklearn.metrics.accuracy_score(test,prediction)
    recall = sklearn.metrics.recall_score(test,prediction)
    precision = sklearn.metrics.precision_score(test,prediction)
    f1 = sklearn.metrics.f1_score(test,prediction)
    jacccard = sklearn.metrics.jaccard_score(test,prediction)
    mean_sqr_err = np.sqrt(mean_squared_error(test, prediction))

    print("Printing metrics...")
    print("mean squared error: {:.2f}%".format(mean_sqr_err*100))
    print("accuracy: {:.2f}%".format(accuracy*100))
    print("recall: {:.2f}%".format(recall*100))
    print("precision: {:.2f}%".format(precision*100))
    print("f1: {:.2f}%".format(f1*100))
    print("jacccard: {:.2f}%".format(jacccard*100))
    print()

def write_metrics_to_file(test, prediction, classifier_name):
    precision, recall, fscore, support = score(test, prediction)
    file_name = '{}{}/metrics.txt'.format(RESULTS_DIR,classifier_name)
    print("Writing metrics to file {}...".format(file_name))
    with open(file_name, 'w') as file:
        file.write('False Precision : {:.2f}\n'.format(precision[0]))
        file.write('False Recall : {:.2f}\n'.format(recall[0]))
        file.write('False fscore : {:.2f}\n\n'.format(fscore[0]))
        file.write('True Precision : {:.2f}\n'.format(precision[1]))
        file.write('True Recall : {:.2f}\n'.format(recall[1]))
        file.write('True fscore : {:.2f}\n'.format(fscore[1]))
    print()

def plot_confusion_matrix(test, prediction, classifier_name, savefig=False):
    print("Plotting confusion matrix for {} Classifier...".format(classifier_name))
    plt.figure()
    conf_matrix = sklearn.metrics.confusion_matrix(test, prediction)
    labels =  np.array([[conf_matrix[0][0],conf_matrix[0][1]],[conf_matrix[1][0],conf_matrix[1][1]]])
    plt.title('Confusion matrix of the {} classifier\n'.format(classifier_name))
    sns.heatmap(conf_matrix, annot=labels, fmt = '', cmap = 'YlGnBu')

    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.gca().xaxis.set_ticklabels(['Real', 'Fake'])
    plt.gca().yaxis.set_ticklabels(['Real', 'Fake'])
    
    if savefig:
        file_name = '{}{}/confusion_matrix.png'.format(RESULTS_DIR,classifier_name)
        print("Saving confusion matrix for {} Classifier to file {}...".format(classifier_name, file_name))
        plt.savefig(file_name)

def plot_feature_importances(X, estimator, classifier_name, savefig=False):
    print("Plotting feature importances for {} Classifier..".format(classifier_name))
    plt.figure()
    feature_importances = pd.Series(estimator.feature_importances_, index=X.columns)
    feature_importances.nlargest(25).plot(kind='barh')
    plt.title("Feature importances for top 25 words")
    if savefig:
        file_name = '{}{}/feature_importances.png'.format(RESULTS_DIR,classifier_name)
        print("Saving feature importances for {} Classifier to file {}...".format(classifier_name, file_name))
        plt.savefig(file_name)