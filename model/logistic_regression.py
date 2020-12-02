import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import sklearn.tree        # For DecisionTreeClassifier class
import sklearn.metrics     # For accuracy_score

import sys
sys.path.insert(1, '../')

import preprocessing.preprocess as preprocess

def plot_data(X, y):
    """Plots a toy 2D data set. Assumes values in range [-3,3] and at most 3 classes."""
    plt.plot(X[y==0,0], X[y==0,1], 'ro', markersize=6)
    plt.plot(X[y==1,0], X[y==1,1], 'bs', markersize=6)
    plt.plot(X[y==2,0], X[y==2,1], 'gx', markersize=6, markeredgewidth=2)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.gca().set_aspect('equal')
    
def plot_predict(model):
    """
    Plots the model's predictions over all points in range 2D [-3, 3].
    If argument is already a Numpy array, treats it as predictions.
    Otherwise calls the argument's predict() function to generate predictions.
    Assumes at most 3 classes.
    """
    extent = (-3, 3, -3, 3)
    x1min, x1max ,x2min, x2max = extent
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 100), np.linspace(x2min, x2max, 100))
    X = np.column_stack([x1.ravel(), x2.ravel()])
    y = model.predict(X).reshape(x1.shape)
    cmap = matplotlib.colors.ListedColormap(['r', 'b', 'g'])
    plt.imshow(y, extent=extent, origin='lower', alpha=0.4, vmin=0, vmax=2, cmap=cmap, interpolation='nearest')
    plt.xlim([x1min, x1max])
    plt.ylim([x2min, x2max])
    plt.gca().set_aspect('equal')
    
def plot_class_probability(model, class_index):
    """
    Plots the model's class probability for the given class {0,1,2}
    over all points in range 2D [-3, 3]. Assumes at most 3 classes.
    """
    extent = (-3, 3, -3, 3)
    x1min, x1max ,x2min, x2max = extent
    x1, x2 = np.meshgrid(np.linspace(x1min, x1max, 100), np.linspace(x2min, x2max, 100))
    X = np.column_stack([x1.ravel(), x2.ravel()])
    p = model.predict_proba(X)[:,class_index].reshape(x1.shape)
    colors = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    cmap = matplotlib.colors.ListedColormap(np.linspace([1, 1, 1], colors[class_index], 50))
    plt.imshow(p, extent=extent, origin='lower', alpha=0.4, vmin=0, vmax=1, cmap=cmap, interpolation='nearest')
    plt.xlim([x1min, x1max])
    plt.ylim([x2min, x2max])
    plt.gca().set_aspect('equal')
    

def decision_tree(X,y, feature_names):
    dtc = sklearn.tree.DecisionTreeClassifier(random_state=0)
    dtc.fit(X,y)

    plot_predict(dtc)
    plot_data(X,y)
    plt.figure()
    sklearn.tree.plot_tree(dtc, feature_names=feature_names)


if __name__ == "__main__":
    fake_news = preprocess.parse_dataset("Fake_test.csv", "FAKE")
    real_news = preprocess.parse_dataset("True_test.csv", "REAL")

    fake_news_all_tokens, fake_news_tokens_per_article = preprocess.tokenize(fake_news, "fake_news")
    real_news_all_tokens, real_news_tokens_per_article = preprocess.tokenize(real_news, "real_news")

    # join tokens and data together
    all_tokens = fake_news_all_tokens + real_news_all_tokens
    tokens_per_article = fake_news_tokens_per_article + real_news_tokens_per_article

    all_news = pd.concat([fake_news, real_news], axis=0)

    # join the data and pass it to split data
    X_train, X_test, y_train, y_test = preprocess.split_and_preprocess(all_tokens,tokens_per_article, all_news)
    
    decision_tree(X_train, y_train, ["FAKE", "REAL"])
