import model.logistic_regression as LR
import model.decision_tree as DT
import model.random_forest as RF
import model.support_vector_machine as SVC
import model.naive_bayesian_classifier as NB
import model.convolutional_neural_network as CNN

import preprocess as preprocess
from tools.ascii_art import logo
from tools.ascii_art import preprocessing
from tools.ascii_art import selection
from tools.ascii_art import divider
from tools.ascii_art import goodbye
from tools.ascii_art import completed

if __name__ == "__main__":
    print("\n" + logo + "\n")
    answer = input("Enter 'm' if you would like to use the mini dataset for testing purposes (results will be inaccurate).\n\nEnter any other key to run full dataset: ")
    
    print("\n" + preprocessing + "\n")
    ml_data, cnn_data  = preprocess.preprocess(answer != 'm')

    X_train, X_test, y_train, y_test = ml_data
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = cnn_data
    preprocess.save_to_csv(X_train, X_test, y_train, y_test, "ml")
    preprocess.save_to_csv(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, "cnn")
    
    while(True):
        print(selection)
        inp = input("\nPlease select the classifier you wish to test. Enter q to exit\n\n0-LogisticRegression\n1-DecisionTree\n2-RandomForest\n3-SVC\n4-NaiveBayes\n5-ConvolutionalNeuralNetwork\n6-RUN ALL MODELS\n")
        
        if inp == '0':
            print(divider)
            LR.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(completed)
        elif inp == '1':
            print(divider)
            DT.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(completed)
        elif inp == '2':
            print(divider)
            RF.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(completed)
        elif inp == '3':
            print(divider)
            SVC.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(completed)
        elif inp == '4':
            print(divider)
            NB.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(completed)
        elif inp == '5':
            CNN.test_run(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, answer != 'm')
        elif inp == '6':
            print(divider)
            LR.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(divider)
            DT.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(divider)
            RF.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(divider)
            SVC.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(divider)
            NB.test_run(X_train, X_test, y_train, y_test, answer != 'm')
            print(divider)
            CNN.test_run(X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn, answer != 'm')
            print(completed)
        elif inp == 'q':
            print(goodbye)
            break
