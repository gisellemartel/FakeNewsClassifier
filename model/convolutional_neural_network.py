''' 
    Source code inspired by implementation here: https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch
'''

import sys
sys.path.insert(1, '../')

import preprocess as preprocess
import tools.tools as tools

#Library imports for deep-learning classifier
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
# warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
np.set_printoptions(precision=3, suppress=True)
torch.manual_seed(0)
np.random.seed(0)

class CnnDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.Tensor(list(x.values))
        self.y = torch.Tensor(list(y.values))
        
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def construct_conv_layers(seq_len, conv_out_size, kernel_sizes, stride):
    layers = []
    for n in kernel_sizes:
        layers.append(nn.Conv1d(seq_len, conv_out_size, n, stride))
    return layers

def construct_max_pooling_layers(kernel_sizes, stride):
    layers = []
    for n in kernel_sizes:
        layers.append(nn.MaxPool1d(n, stride))
    return layers

def calculate_output_size(layer_out_size, kernel_size, stride):
    return math.floor(((layer_out_size - 1 * (kernel_size - 1) - 1) /stride ) + 1)

def get_fully_connected_layer_input(embedding_size, kernel_sizes, stride):
    fully_connected_layer_input = 0
    
    # Determine size of conv and pool features for each layer
    for kernel_size in kernel_sizes:
        conv_out = calculate_output_size(embedding_size, kernel_size, stride)
        pool_out = calculate_output_size(conv_out, kernel_size, stride)
        fully_connected_layer_input = fully_connected_layer_input + pool_out
    
    return fully_connected_layer_input

def build_cnn_layers(params):
    layers = {}
    seq_len = params["seq_len"]
    num_words = params["num_words"]
    conv_out_size = params["conv_out_size"]
    stride = params["stride"]
    kernel_sizes = params["kernel_sizes"]
    embedding_size = params["embedding_size"]

    layers["embedding"] = nn.Embedding(num_words + 1, embedding_size, padding_idx=0)
    
    layers["conv_layers"] = construct_conv_layers(seq_len, conv_out_size, kernel_sizes, stride)
    layers["pool_layers"] = construct_max_pooling_layers(kernel_sizes, stride)
    
    input = get_fully_connected_layer_input(embedding_size,kernel_sizes,stride) * conv_out_size
    layers["fully_connected"] = nn.Linear(input, 1)

    return layers

def apply_conv_layers(conv_layers, pool_layers, x, N):
    X_ = []
    for i in range(N):
        x_ = conv_layers[i](x)
        x_ = torch.relu(x_)
        x_ = pool_layers[i](x_)
        X_.append(x_)
    return X_

class CnnModel(nn.ModuleList):	

    def __init__(self, params):
        super(CnnModel, self).__init__()

        self.dropout = nn.Dropout(0.25)
        self.layers = build_cnn_layers(params)

        # parse the layers as params for the model
        for layer in self.layers:
            item = self.layers[layer]
            if type(item) == list:
                i = 1
                for l in item:
                    name = "{}_{}".format(layer, i)
                    setattr(self, name, l)
                    i = i + 1
            else:
                setattr(self,layer, item)
        
    def forward(self, x):
        # Sequence of tokes is filterd through an embedding layer
        x = torch.tensor(x).to(torch.int64)
        x = self.layers["embedding"](x)

        assert(len(self.layers["pool_layers"]) == len(self.layers["conv_layers"]))
        N = len(self.layers["conv_layers"])

        # get output of each conv layer
        x1, x2, x3, x4 = apply_conv_layers(self.layers["conv_layers"], self.layers["pool_layers"], x, N)
        
        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2)
        union = union.reshape(union.size(0), -1)

        # The "flattened" vector is passed through a fully connected layer
        out = self.layers["fully_connected"](union)
        # Dropout is applied		
        out = self.dropout(out)
        
        # apply the activation function
        out = torch.sigmoid(out)
        
        return out.squeeze()

def process_batch(model, optimizer, x_batch, y_batch):
    x_batch = torch.tensor(x_batch).to(torch.int64)
    y_batch = y_batch.type(torch.Tensor)

    # ensure the shape for batches is the same
    x_batch_shape = list(x_batch.size())[0]
    y_batch = torch.reshape(y_batch, (x_batch_shape,))

    # Get the prediction for the current batch
    y_pred = model(x_batch)

    # calculate the loss
    loss = F.binary_cross_entropy(y_pred, y_batch)

    # Calculate the gradient and update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() 

    predictions = list(y_pred.detach().numpy())

    return predictions, loss

def generate_batch_data(X,y,batch_size):
    return DataLoader(CnnDataset(X, y), batch_size=batch_size)

def train_cnn(model, X_train, X_test, y_train, y_test, epochs, batch_size, learning_rate):
    loader_train = generate_batch_data(X_train,y_train,batch_size)
    loader_test = generate_batch_data(X_test,y_test,batch_size)

    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    # carry out training of model
    for epoch in range(epochs):

        model.train()
        predictions = []
        
        training_loss = 0
        # feed the batches to the neural network
        for x_batch, y_batch in loader_train:
            pr, batch_loss = process_batch(model, optimizer, x_batch, y_batch)
            predictions += pr
            training_loss = training_loss + batch_loss
        
        train_losses.append(training_loss.item())
        
        # Evaluate the prediction result
        test_predictions, test_loss = predict(model, loader_test)
        test_losses.append(test_loss.item())
        
        # Metrics calculation
        train_accuracy = calculate_estimation_accuracy(y_train.values, predictions)*100
        test_accuracy = calculate_estimation_accuracy(y_test.values, test_predictions)*100

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        print("Epoch: {}, loss: {:.5f}, Train accuracy: {:.5f}%, Test accuracy: {:.5f}%".format(epoch+1, training_loss.item(), train_accuracy, test_accuracy))

    return train_accuracies, test_accuracies, train_losses, test_losses

def predict(model, batch_data):
    # Set the model in evaluation mode
    model.eval()
    predictions = []
    loss = 0

    # evaluate predictions in current epoch
    with torch.no_grad():
        for x_batch, y_batch in batch_data:
            # ensure the shape for batches is the same
            x_batch_shape = list(x_batch.size())[0]
            y_batch = torch.reshape(y_batch, (x_batch_shape,))

            y_pred = model(x_batch)

            predictions += list(y_pred.detach().numpy())

            loss += F.binary_cross_entropy(y_pred, y_batch)

    return predictions, loss
    
def calculate_estimation_accuracy(true_values, predictions):
    # Metrics calculation
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(true_values, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives+true_negatives) / len(true_values)


def test_run(X_train, X_test, y_train, y_test, use_full_dataset=False):
    if(not use_full_dataset) : tools.set_results_dir("./results/mock_results/")
    print("Testing Convolutional Neural Network Classifier ...\n")

    # params for CNN model
    seq_len = 219
    if use_full_dataset: 
        seq_len = 216
        
    model_params = {
        # text preprocessing
        "seq_len": seq_len,
        "num_words": 10000,
        "embedding_size": 64,

        # size of convolution outputs
        "conv_out_size": 32,

        # Number of strides for each convolution
        "stride": 2,

        # kernel sizes
        "kernel_sizes": [2,3,4,5]
    }

    # training parameters
    epochs = 24
    batch_size = 108
    learning_rate = 0.0001

    model = CnnModel(model_params)
    train_accuracies, test_accuracies, train_losses, test_losses \
        = train_cnn(
            model, 
            X_train, 
            X_test, 
            y_train, 
            y_test, 
            epochs, 
            batch_size, 
            learning_rate
        )

    tools.plot_cnn_accuracies(train_accuracies,test_accuracies, "CNN", epochs, batch_size, learning_rate, True)
    tools.plot_cnn_losses(train_losses, test_losses, "CNN", epochs, batch_size, learning_rate, True)

    batch_data = generate_batch_data(X_test,y_test,batch_size)
    y_pred, loss = predict(model, batch_data)

    y_pred_labels = tools.calculate_neural_net_predicted_labels(y_pred)
    accuracy = tools.calculate_neural_net_accuracy(y_test.values, y_pred)
    print("\nConvolutional Neural Network prediction accuracy: {:.5f}%".format(accuracy*100))
    print("Convolutional Neural Network prediction loss: {:.5f}\n".format(loss.item()))

    tools.plot_predicted_labels(y_test.values, y_pred_labels, "CNN", True)
    tools.display_prediction_scores(y_test.values,y_pred_labels)
    tools.write_metrics_to_file(y_test.values,y_pred_labels,"CNN")
    tools.plot_confusion_matrix(y_test.values,y_pred_labels,"CNN", True)
