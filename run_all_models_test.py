import model.logistic_regression as LR
import model.decision_tree as DT
import model.random_forest as RF
import model.support_vector_machine as SVC
import model.naive_bayesian_classifier as NB
import model.convolutional_neural_network as CNN

import preprocess as preprocess

logo = '''___            ___          ___       __      __   ___ ___  ___  __  ___  __   __  
|__   /\  |__/ |__     |\ | |__  |  | /__`    |  \ |__   |  |__  /  `  |  /  \ |__) 
|    /~~\ |  \ |___    | \| |___ |/\| .__/    |__/ |___  |  |___ \__,  |  \__/ |  \ 
                                                                                    '''

preprocessing = ''' __   __   ___  __   __   __   __   ___  __   __          __      __       ___         
|__) |__) |__  |__) |__) /  \ /  ` |__  /__` /__` | |\ | / _`    |  \  /\   |   /\     
|    |  \ |___ |    |  \ \__/ \__, |___ .__/ .__/ | | \| \__>    |__/ /~~\  |  /~~\ ...
                                                                                       '''

selection = '''
 __             __   __     ___    ___  __      __   ___       ___  __  ___    __        
/  ` |     /\  /__` /__` | |__  | |__  |__)    /__` |__  |    |__  /  `  |  | /  \ |\ | .
\__, |___ /~~\ .__/ .__/ | |    | |___ |  \    .__/ |___ |___ |___ \__,  |  | \__/ | \| .
                                                                                         
                                                                                         '''

divider = '''
 __             __   __     ___             __           ___       __     
/  ` |     /\  /__` /__` | |__  \ / | |\ | / _`    |\ | |__  |  | /__`    
\__, |___ /~~\ .__/ .__/ | |     |  | | \| \__>    | \| |___ |/\| .__/ ...
                                                                          
                                                                          '''

goodbye='''
 __   __   __   __   __       ___ 
/ _` /  \ /  \ |  \ |__) \ / |__  
\__> \__/ \__/ |__/ |__)  |  |___ 
                                  
                                  '''

if __name__ == "__main__":
    print("\n" + logo + "\n")
    answer = input("Enter 'm' if you would like to use the mini dataset for testing purposes (results will be inaccurate).\n\nEnter any other key to run full dataset: ")
    
    print("\n" + preprocessing + "\n")
    X_train, X_test, y_train, y_test, all_tokens = preprocess.preprocess_test(answer != 'm')
    
    while(True):
        print(selection)
        inp = input("\nPlease select the classifier you wish to test. Enter q to exit\n\n0-LogisticRegression\n1-DecisionTree\n2-RandomForest\n3-SVC\n4-NaiveBayes\n5-ConvolutionalNeuralNetwork\n6-RUN ALL MODELS\n")
        
        if inp == '0':
            print(divider)
            LR.test_run(X_train, X_test, y_train, y_test)
        elif inp == '1':
            print(divider)
            DT.test_run(X_train, X_test, y_train, y_test)
        elif inp == '2':
            print(divider)
            RF.test_run(X_train, X_test, y_train, y_test)
        elif inp == '3':
            print(divider)
            SVC.test_run(X_train, X_test, y_train, y_test)
        elif inp == '4':
            print(divider)
            NB.test_run(X_train, X_test, y_train, y_test)
        # elif inp == '5':
        #     CNN.test_run(X_train, X_test, y_train, y_test)
        elif inp == '6':
            print(divider)
            LR.test_run(X_train, X_test, y_train, y_test)
            print(divider)
            DT.test_run(X_train, X_test, y_train, y_test)
            print(divider)
            RF.test_run(X_train, X_test, y_train, y_test)
            print(divider)
            SVC.test_run(X_train, X_test, y_train, y_test)
            print(divider)
            NB.test_run(X_train, X_test, y_train, y_test)
        elif inp == 'q':
            print(goodbye)
            break
