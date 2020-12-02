import sklearn
import sklearn.tree

import sys
sys.path.insert(1, '../')

import preprocessing.preprocess as preprocess
import tools as tools

def decision_tree_train(X, y):
    model = sklearn.tree.DecisionTreeClassifier(random_state=0)
    model.fit(X,y)
    return model

def decision_tree_predict(model, X):
    return model.predict(X)

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test()

    print("\nTesting Decision Tree Classifier ...\n")

    model = decision_tree_train(X_train, y_train)
    y_pred = decision_tree_predict(model, X_test)

    tools.display_prediction_scores(y_test,y_pred)

