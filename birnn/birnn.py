import torch
import torch.nn as nn
from scripts import dataset_birnn
import numpy as np

import time

# Dataset paths
train_path = f'birnn/train.pt'
test_path = f'birnn/test.pt'

# Model hyperparams
n_layers = 1
hidden_size = 16
learning_rate = 0.001
n_epochs = 50

# DataLoader hyperparams
train_loader_params = {'batch_size': 32,
                       'shuffle': True,
                       'num_workers': 20}

# If you change this you should modify the build_dataset script accordingly
# Length of each byte sequence sample
seq_len = 1000


class FunctionBoundsModel(nn.Module):
    """Bidirectional RNN for predicting function starts/ends in byte sequences. """

    def __init__(self, hidden_size, n_layers, seq_len):
        super(FunctionBoundsModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # First argument is 1 b/c the features of a sample at each time step is just a single byte
        # Output shape of RNN layer is (batch_size, seq_len, hidden_size*2)
        self.rnn = nn.RNN(1, hidden_size, n_layers, batch_first=True, bidirectional=True)

        # Output shape of FC layer is (batch_size, seq_len, n_classes=3)
        self.fc = nn.Linear(hidden_size * 2, 3)

    def forward(self, X):
        # Set initial state of hidden layer
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        h_0 = torch.zeros(self.n_layers * 2, X.shape[0], self.hidden_size).to(device)

        out, _ = self.rnn(X, h_0)
        out = self.fc(out)

        return out


def train_model():
    """Train model using training dataset and hyperparameters specified in global variables.
    
    Returns:
        FunctionBoundsModel: trained bidirectional RNN model.
        
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_set = dataset_birnn.FunctionBoundsDataset(train_path)
    train_loader = torch.utils.data.DataLoader(train_set, **train_loader_params)

    model = FunctionBoundsModel(hidden_size, n_layers, seq_len).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(n_epochs):

        total_loss = 0

        for i, (X, Y) in enumerate(train_loader):
            X = X.to(device)
            Y = Y.to(device)

            # Forward pass
            outputs = model(X)

            # Flatten the sequence dimension to simplify computing loss
            outputs = outputs.view(-1, 3)
            Y = Y.view(-1)

            loss = criterion(outputs, Y)
            total_loss += loss.item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print running loss
            if i % 8 == 7:
                print("Epoch [{}/{}], Step [{}/{}], Loss: {:.6f}"
                      .format(epoch + 1, n_epochs, i + 1, len(train_loader), total_loss / 1000))
                total_loss = 0

        model.eval()
        eval_model(model)
        model.train()

    return model


def eval_model(model):
    """Evaluates model performance on testing dataset; computes confusion matrix and corresponding
    F1 score, precision and recall.
    
    Args:
        model(FunctionBoundsModel): trained bidirectional RNN for function starts/ends prediction.
        
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_set = dataset_birnn.FunctionBoundsDataset(test_path)
    test_loader = torch.utils.data.DataLoader(test_set, **train_loader_params)

    with torch.no_grad():

        byte_count = 0

        # Confusion matrix
        cm = np.zeros((3, 3), dtype=np.long)

        eval_time = 0
        stats_time = 0

        for i, (X, Y) in enumerate(test_loader):

            eval_start = time.time()

            X = X.to(device)
            Y = Y.to(device)

            outputs = model(X)

            # Flatten the sequence dimension to simplify computing loss
            outputs = outputs.view(-1, 3)
            Y = Y.view(-1)

            # Update stats
            _, preds = torch.max(outputs, 1)

            eval_time += time.time() - eval_start

            stats_start = time.time()

            byte_count += X.shape[0] * X.shape[1]
            _, preds = torch.max(outputs, 1)

            # Update confusion matrix; 0 = not function start/end; 1 = start; 2 = end
            for pred, y in zip(preds.cpu().numpy(), Y.cpu().numpy()):
                cm[pred, y] += 1

        f_starts_precision = cm[1, 1].item() / ((cm[1, 0] + cm[1, 1] + cm[1, 2]).item() + 1e-5)
        f_starts_recall = cm[1, 1].item() / ((cm[0, 1] + cm[1, 1] + cm[2, 1]).item() + 1e-5)

        # Prevent div by 0 in case low precision and recall
        if f_starts_precision + f_starts_recall == 0:
            f_starts_f1 = 0
        else:
            f_starts_f1 = 2 * ((f_starts_precision * f_starts_recall) / ((f_starts_precision + f_starts_recall) + 1e-5))

        f_ends_precision = cm[2, 2].item() / ((cm[2, 0] + cm[2, 1] + cm[2, 2]).item() + 1e-5)
        f_ends_recall = cm[2, 2].item() / ((cm[0, 2] + cm[1, 2] + cm[2, 2]).item() + 1e-5)

        if f_ends_precision + f_ends_recall == 0:
            f_ends_f1 = 0
        else:
            f_ends_f1 = 2 * ((f_ends_precision * f_ends_recall) / ((f_ends_precision + f_ends_recall) + 1e-5))

        print()
        print("Function starts prediction metrics:")
        print("F1: {:.6f}, Precision: {:.6f}, Recall: {:.6f}"
              .format(f_starts_f1, f_starts_precision, f_starts_recall))
        print()
        print("Function ends prediction metrics:")
        print("F1: {:.6f}, Precision: {:.6f}, Recall: {:.6f}"
              .format(f_ends_f1, f_ends_precision, f_ends_recall))
        print()
        print(cm)
        print()

        byte_count = 0
        cm = np.zeros((3, 3), dtype=np.long)


if __name__ == "__main__":
    model = train_model()
