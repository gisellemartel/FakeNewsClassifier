import model.logistic_regression as LR
import model.decision_tree as DT
import model.random_forest as RF
import model.support_vector_machine as SVC
import model.naive_bayesian_classifier as NB

import preprocess as preprocess

if __name__ == "__main__":
    answer = input("Enter 'm' if you would like to use the mini dataset for testing purposes (results will be inaccurate). Enter any other key to run full dataset: ")
    
    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test(answer != 'm')
    
    while(True):
        inp = input("\nPlease select the classifier you wish to test. Enter q to exit\n0-LogisticRegression\n1-DecisionTree\n2-RandomForest\n3-SVC\n4-NaiveBayes ")
        
        if inp == '0':
            LR.test_run(X_train, X_test, y_train, y_test)
        elif inp == '1':
            DT.test_run(X_train, X_test, y_train, y_test)
        elif inp == '2':
            RF.test_run(X_train, X_test, y_train, y_test)
        elif inp == '3':
            SVC.test_run(X_train, X_test, y_train, y_test)
        elif inp == '4':
            NB.test_run(X_train, X_test, y_train, y_test)
        elif inp == 'q':
            break
