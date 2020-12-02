from sklearn.svm import SVC

import sys
sys.path.insert(1, '../')

import preprocessing.preprocess as preprocess
import tools as tools

def support_vector_machine_train(X,y):
    print("Fitting data to SVC, this may take a while...")
    svc = SVC(kernel='linear', random_state=0)
    svc.fit(X, y)
    return svc

def support_vector_machine_predict(svc, X):
    return svc.predict(X)

if __name__ == "__main__":
    preprocess.override_dir("../")
    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test()

    print("\nTesting SVM Classifier ...\n")

    model = support_vector_machine_train(X_train, y_train)
    y_pred = support_vector_machine_predict(model, X_test)

    tools.display_prediction_scores(y_test,y_pred)
    tools.write_metrics_to_file(y_test,y_pred,"SVC")
    tools.plot_confusion_matrix(y_test,y_pred,"SVC", True)

