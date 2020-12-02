import sklearn
from sklearn.naive_bayes import MultinomialNB

import sys
sys.path.insert(1, '../')

import preprocess as preprocess
import tools as tools

def naive_bayesian_train(X,y):
    model = MultinomialNB()
    model.fit(X, y)
    return model

def naive_bayesian_predict(model, X):
    return model.predict(X)

def display_result(model, X_train):
    #Storing the number of times each token occurs in a True article
    true_token_count = model.feature_count_[0, :]

    #Storing the number of times each token appears in a Fake article
    fake_token_count = model.feature_count_[1, :]

    #create a dataframe out of the new data
    tokens = pd.DataFrame({'token':X_train.columns, 'true':true_token_count, 'fake':fake_token_count}).set_index('token')
    tokens['true'] = tokens.true + 1 #avoid division by 0 when doing frequency calculations
    tokens['fake'] = tokens.fake + 1
    tokens['true'] = tokens.true / model.class_count_[0]
    tokens['fake'] = tokens.fake / model.class_count_[1]

    tokens['fake/true ratio'] = tokens.fake / tokens.true
    tokens.sort_values('fake/true ratio', ascending=False).head(10)

def test_run():
    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test()

    print("\nTesting Naive Bayesian Classifier ...\n")

    model = naive_bayesian_train(X_train, y_train)
    y_pred = naive_bayesian_predict(model, X_test)

    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"NaiveBayes")
    tools.plot_confusion_matrix(y_test,y_pred,"NaiveBayes", True)

if __name__ == "__main__":
    test_run()

