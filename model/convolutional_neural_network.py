import sys
sys.path.insert(1, '../')

import preprocess as preprocess
import tools as tools

#Library imports for deep-learning classifier
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
np.set_printoptions(precision=3, suppress=True)

def in_features_fc(params):
    '''Calculates the number of output features after Convolution + Max pooling
        
    Convolved_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    Pooled_Features = ((embedding_size + (2 * padding) - dilation * (kernel - 1) - 1) / stride) + 1
    
    source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    '''
    # Calcualte size of convolved/pooled features for convolution_1/max_pooling_1 features
    seq_len, num_words, embedding_size, stride, k1, k2, k3, k4, out_size, hyperparameters = params
    out_conv_1 = ((embedding_size - 1 * (k1 - 1) - 1) / stride) + 1
    out_conv_1 = math.floor(out_conv_1)
    out_pool_1 = ((out_conv_1 - 1 * (k1 - 1) - 1) / stride) + 1
    out_pool_1 = math.floor(out_pool_1)
    
    # Calcualte size of convolved/pooled features for convolution_2/max_pooling_2 features
    out_conv_2 = ((embedding_size - 1 * (k2 - 1) - 1) / stride) + 1
    out_conv_2 = math.floor(out_conv_2)
    out_pool_2 = ((out_conv_2 - 1 * (k2 - 1) - 1) / stride) + 1
    out_pool_2 = math.floor(out_pool_2)
    
    # Calcualte size of convolved/pooled features for convolution_3/max_pooling_3 features
    out_conv_3 = ((embedding_size - 1 * (k3 - 1) - 1) / stride) + 1
    out_conv_3 = math.floor(out_conv_3)
    out_pool_3 = ((out_conv_3 - 1 * (k3 - 1) - 1) / stride) + 1
    out_pool_3 = math.floor(out_pool_3)
    
    # Calcualte size of convolved/pooled features for convolution_4/max_pooling_4 features
    out_conv_4 = ((embedding_size - 1 * (k4 - 1) - 1) / stride) + 1
    out_conv_4 = math.floor(out_conv_4)
    out_pool_4 = ((out_conv_4 - 1 * (k4 - 1) - 1) / stride) + 1
    out_pool_4 = math.floor(out_pool_4)
    
    # Returns "flattened" vector (input for fully connected layer)
    return (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * out_size
    
def build_cnn_model(params):
    seq_len, num_words, embedding_size, stride, k1, k2, k3, k4, out_size, hyperparameters = params 
    # Dropout definition
    dropout = nn.Dropout(0.25)
    
    # Embedding layer definition
    embedding = nn.Embedding(num_words + 1, embedding_size, padding_idx=0)
    
    # Convolution layers definition
    conv_1 = nn.Conv1d(seq_len, out_size, k1, stride)
    conv_2 = nn.Conv1d(seq_len, out_size, k2, stride)
    conv_3 = nn.Conv1d(seq_len, out_size, k3, stride)
    conv_4 = nn.Conv1d(seq_len, out_size, k4, stride)
    
    # Max pooling layers definition
    pool_1 = nn.MaxPool1d(k1, stride)
    pool_2 = nn.MaxPool1d(k2, stride)
    pool_3 = nn.MaxPool1d(k3, stride)
    pool_4 = nn.MaxPool1d(k4, stride)
    
    # Fully connected layer definition
    fc = nn.Linear(in_features_fc(params), 1)

def forward(self, x):

    # Sequence of tokes is filterd through an embedding layer
    x = self.embedding(x)
    
    # Convolution layer 1 is applied
    x1 = self.conv_1(x)
    x1 = torch.relu(x1)
    x1 = self.pool_1(x1)
    
    # Convolution layer 2 is applied
    x2 = self.conv_2(x)
    x2 = torch.relu((x2))
    x2 = self.pool_2(x2)

    # Convolution layer 3 is applied
    x3 = self.conv_3(x)
    x3 = torch.relu(x3)
    x3 = self.pool_3(x3)
    
    # Convolution layer 4 is applied
    x4 = self.conv_4(x)
    x4 = torch.relu(x4)
    x4 = self.pool_4(x4)
    
    # The output of each convolutional layer is concatenated into a unique vector
    union = torch.cat((x1, x2, x3, x4), 2)
    union = union.reshape(union.size(0), -1)

    # The "flattened" vector is passed through a fully connected layer
    out = self.fc(union)
    # Dropout is applied		
    out = self.dropout(out)
    # Activation function is applied
    out = torch.sigmoid(out)
    
    return out.squeeze()

def train(model, X_train, X_test, y_train, y_test, vocabulary, params):
    
    # # Initialize dataset maper
    # train = DatasetMaper(data['x_train'], data['y_train'])
    # test = DatasetMaper(data['x_test'], data['y_test'])
    
    # # Initialize loaders
    # loader_train = DataLoader(train, batch_size=params.batch_size)
    # loader_test = DataLoader(test, batch_size=params.batch_size)
    
    # # Define optimizer
    # optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
    
    # Starts training phase
    for epoch in range(params.epochs):
        # Set model in training model
        model.train()
        predictions = []
        # Starts batch training
        for x_batch, y_batch in loader_train:
        
            y_batch = y_batch.type(torch.FloatTensor)
            
            # Feed the model
            y_pred = model(x_batch)
            
            # Loss calculation
            loss = F.binary_cross_entropy(y_pred, y_batch)
            
            # Clean gradientes
            optimizer.zero_grad()
            
            # Gradients calculation
            loss.backward()
            
            # Gradients update
            optimizer.step()
            
            # Save predictions
            predictions += list(y_pred.detach().numpy())
        
        # Evaluation phase
        test_predictions = evaluation(model, loader_test)
        
        # Metrics calculation
        train_accuary = calculate_accuray(y_train, predictions)
        test_accuracy = calculate_accuray(y_test, test_predictions)
        print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))
        
def evaluation(model, loader_test):
    # Set the model in evaluation mode
    model.eval()
    predictions = []
    
    # Starst evaluation phase
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            y_pred = model(x_batch)
            predictions += list(y_pred.detach().numpy())
    return predictions
    
def calculate_accuray(grand_truth, predictions):
    # Metrics calculation
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(grand_truth, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives+true_negatives) / len(grand_truth)


def test_run(X_train, X_test, y_train, y_test, vocabulary, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./test_results/")
    print("Testing Convolutional Neural Network Classifier ...\n")

    #TODO: dont hardcode seq len
    seq_len = 35
    num_words = len(vocabulary)
    embedding_size = 64
    # Number of strides for each convolution
    stride = 2
    # kernel sizes
    k1 = 2
    k2 = 3
    k3 = 4
    k4 = 5
    # output size for each convolution
    out_size = 32
    
    # training hyperparams
    epochs = 10
    batch_sizev = 12
    learning_rate = 0.001

    hyperparameters = {"epochs": epochs, "batch_sizev": batch_sizev, "learning_rate": learning_rate}

    params = [seq_len, num_words, embedding_size, stride, k1, k2, k3, k4, out_size, hyperparameters]

    # Initialize the model
    model = build_cnn_model(params)
    
    # Training - Evaluation pipeline
    # train(model, X_train, X_test, y_train, y_test, vocabulary, params)