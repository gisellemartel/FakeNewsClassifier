import warnings
warnings.filterwarnings("ignore")

import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sklearn
import sklearn.metrics
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import precision_recall_fscore_support as score

RESULTS_DIR = "./results/"

'''
    Generates a pie chart of the label distribution
'''
def plot_labels(y):
    labels = 'Real news','Fake news'
    sizes = [len(y[y==0]), len(y[y==1])]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.gca().axis('equal')

'''
    Generates a pie charts of true vs. predicted label distributions
'''
def plot_predicted_labels(test, prediction, classifier_name, savefig=False):
    plt.figure()
    fig, ax = plt.subplots(nrows=1, ncols=2)

    ax[0].set_title("True Labels")
    plt.sca(ax[0])
    plot_labels(test)
        
    plt.sca(ax[1])
    ax[1].set_title("Predicted Labels")
    plot_labels(prediction)

    fig.set_size_inches((11,4))

    if savefig:
        file_name = '{}{}/piechart.png'.format(RESULTS_DIR,classifier_name)
        print("Saving piechart of results for {} Classifier to file {}...".format(classifier_name, file_name))
        plt.savefig(file_name)

'''
    Displays detail of best estimator
'''
def display_best_estimator(accuracy, estimator, hyperparams):
    print("Best {} estimator accuracy: {:.3f}%".format(estimator, accuracy))
    print("Hyperparams:")

    for param in hyperparams:
        print("{}: {}".format(param, hyperparams[param]))

    print()

'''
    Displays the metrics of the model prediction
'''
def display_prediction_scores(test,prediction):
    accuracy = sklearn.metrics.accuracy_score(test,prediction)
    recall = sklearn.metrics.recall_score(test,prediction)
    precision = sklearn.metrics.precision_score(test,prediction)
    f1 = sklearn.metrics.f1_score(test,prediction)
    jacccard = sklearn.metrics.jaccard_score(test,prediction)
    mean_sqr_err = np.sqrt(mean_squared_error(test, prediction))

    print("Printing metrics...")
    print("mean squared error: {:.3f}%".format(mean_sqr_err*100))
    print("accuracy: {:.3f}%".format(accuracy*100))
    print("recall: {:.3f}%".format(recall*100))
    print("precision: {:.3f}%".format(precision*100))
    print("f1: {:.3f}%".format(f1*100))
    print("jacccard: {:.3f}%".format(jacccard*100))
    print()

'''
    Stores metrics of a given model to a .txt file
'''
def write_metrics_to_file(test, prediction, classifier_name):
    precision, recall, fscore, support = score(test, prediction)
    file_name = '{}{}/metrics.txt'.format(RESULTS_DIR,classifier_name)
    print("Writing metrics to file {}...".format(file_name))
    with open(file_name, 'w') as file:
        file.write('False Precision : {:.3f}\n'.format(precision[0]))
        file.write('False Recall : {:.3f}\n'.format(recall[0]))
        file.write('False fscore : {:.3f}\n\n'.format(fscore[0]))
        file.write('True Precision : {:.3f}\n'.format(precision[1]))
        file.write('True Recall : {:.3f}\n'.format(recall[1]))
        file.write('True fscore : {:.3f}\n'.format(fscore[1]))
    print()

'''
    plots the prediction vs. true labels of a model as a confusion matrix
'''
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

'''
    Plots the feature importances of an estimator
'''
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

'''
    Determines the overfitting of a given model based on its estimator scores
'''
def determine_overfitting(trn_scores, test_scores):
    # calculate the overall absolute difference between the test and training results for each classifier 
    overfitting = 0
    for i in range(len(trn_scores)):
        overfitting = overfitting + abs(trn_scores[i] - test_scores[i]) 
    return overfitting

'''
    Calculates training and testing scores for estimators
'''
def calculate_estimator_scores(data, estimators):
    X_train, X_test, y_train, y_test = data

    def score_estimators(X, y, estimators):
        scores = []
        for e in estimators: scores.append(e.score(X,y))
        return scores

    trn_scores = score_estimators(X_train, y_train, estimators)
    test_scores = score_estimators(X_test, y_test, estimators)

    return trn_scores, test_scores

''' plots the scores for each estimator of a given model
    code in this function inspired by Assignment 1
    Helps visualize performance of different hyperparams 
'''
def plot_estimator_scores(name,trn_scores,test_scores,savefig=False):

    test_score, trn_score = np.max(test_scores),np.max(trn_scores)

    plt.figure()
    x = np.linspace(0, len(trn_scores), len(trn_scores))
    
    plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
    plt.ylabel("score")
    
    plt.text(40, 0.1, "train = {:.3f}".format(trn_score), fontsize=10, c="blue")
    plt.text(40, 0.2, "test = {:.3f}".format(test_score), fontsize=10, c='purple')

    plt.plot(x,trn_scores, label="train", c="blue", marker="o")
    plt.plot(x,test_scores, label="test", c="purple",  marker="o")

    xpos = test_scores.index(test_score)
    xpos = x[xpos]

    plt.plot(xpos, test_score, c='r', marker='x', ms=15)

    plt.ylim([0,1.1])
    
    plt.title( "{} Scores for all Hyperparameter Combinations".format(name))
    plt.legend()

    if savefig:
        file_name = '{}{}/scores_plot.png'.format(RESULTS_DIR,name)
        print("Saving estimator scores for {} Classifier to file {}...".format(name, file_name))
        plt.savefig(file_name)