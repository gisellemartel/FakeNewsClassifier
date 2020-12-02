import sklearn

import sys
sys.path.insert(1, '../')

import preprocess as preprocess
import tools as tools

def logistic_regression_train(X, y):
    model = sklearn.linear_model.LogisticRegression(random_state=0)
    model.fit(X,y)
    return model

def logistic_regression_predict(model, X):
    return model.predict(X)

def test_run():
    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test()

    print("\nTesting Logistic Regression Classifier ...\n")

    model = logistic_regression_train(X_train, y_train)
    y_pred = logistic_regression_predict(model, X_test)

    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"LogisticRegression")
    tools.plot_confusion_matrix(y_test,y_pred,"LogisticRegression", True)

if __name__ == "__main__":
    test_run()


