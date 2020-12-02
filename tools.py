import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.metrics 

def display_prediction_scores(test,prediction):
    accuracy = sklearn.metrics.accuracy_score(test,prediction)
    recall = sklearn.metrics.recall_score(test,prediction)
    precision = sklearn.metrics.precision_score(test,prediction)
    f1 = sklearn.metrics.f1_score(test,prediction)
    jacccard = sklearn.metrics.jaccard_score(test,prediction)
    
    print("accuracy: {:.2f}%".format(accuracy*100))
    print("recall: {:.2f}%".format(recall*100))
    print("precision: {:.2f}%".format(precision*100))
    print("f1: {:.2f}%".format(f1*100))
    print("jacccard: {:.2f}%".format(jacccard*100))