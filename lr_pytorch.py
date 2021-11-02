# Developed from Code from Alex Jian Zheng
import random
import math
import torch
import torch.nn as nn
import numpy as np
from numpy import zeros, sign
from math import exp, log
from collections import defaultdict
import json

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import argparse

from sgd import Example

torch.manual_seed(1701)

class GuessDataset(Dataset):
    def __init__(self, vocab):
        self.vocab = vocab
        
        # Just create some dummy data so unit tests will fail rather than cause error
        self.num_features = len(self.vocab)
        self.feature = zeros((5, self.num_features))
        self.label = zeros((5, 1))
        self.num_samples = 5

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.feature[index], self.label[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.num_samples

    def initialize(self, filename):
        # Complete this function to actually populate the feature and label members of the class with non-zero data.
        dataset = []                                                        #sol
        with open(filename) as infile:                                      #sol
            for line in infile:                                             #sol
                ex = Example(json.loads(line), self.vocab, use_bias=False)  #sol
                dataset.append(ex)                                          #sol

        # You may want to use numpy's fromiter function
        
        features = np.stack(list(ex.x for ex in dataset))                   #sol
        label = np.stack(list(np.array([ex.y]) for ex in dataset))          #sol

        self.feature = torch.from_numpy(features.astype(np.float32))        #sol
        self.label = torch.from_numpy(label.astype(np.float32))             #sol
        self.num_samples = len(self.label)                                  #sol
        assert self.num_samples == len(self.feature)
        None

class SimpleLogreg(nn.Module):
    def __init__(self, num_features):
        """
        Initialize the parameters you'll need for the model.

        :param num_features: The number of features in the linear model
        """
        super(SimpleLogreg, self).__init__()
        self.linear = nn.Linear(num_features, 1)        #sol

    def forward(self, x):
        """
        Compute the model prediction for an example.

        :param x: Example to evaluate
        """
        y_pred = torch.sigmoid(self.linear(x))          #sol Label has only two categories, so sigmoid and softmax should be essentially the same.
        return y_pred                                   #sol

    def evaluate(self, data):
        """
        Computes the accuracy of the model. 
        """
        # No need to modify this function.
        with torch.no_grad():
            y_predicted = self(data.feature)
            y_predicted_cls = y_predicted.round()
            acc = y_predicted_cls.eq(data.label).sum() / float(data.label.shape[0])
            return acc

def step(epoch, ex, model, optimizer, criterion, inputs, labels):
    """Take a single step of the optimizer, we factored it into a single
    function so we could write tests.  You should: A) get predictions B)
    compute the loss from that prediction C) backprop D) update the
    parameters

    There's additional code to print updates (for good software
    engineering practices, this should probably be logging, but printing
    is good enough for a homework).

    :param epoch: The current epoch
    :param ex: Which example / minibatch you're one
    :param model: The model you're optimizing
    :param inputs: The current set of inputs
    :param labels: The labels for those inputs
    """
    y_pred = model(inputs)                                  #sol
    loss = criterion(y_pred, labels)                        #sol
    loss.backward()                                         #sol
    optimizer.step()                                        #sol

    optimizer.zero_grad()                                   #sol
    if (ex+1) % 20 == 0:
      acc_train = model.evaluate(train)
      acc_test = model.evaluate(test)
      print(f'Epoch: {epoch+1}/{num_epochs}, Example {ex}, loss = {loss.item():.4f}, train_acc = {acc_train.item():.4f} test_acc = {acc_test.item():.4f}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    #''' Switch between the toy and REAL EXAMPLES
    argparser.add_argument("--buzztrain", help="Positive class",
                           type=str, default="small_guess.buzztrain.jsonl")
    argparser.add_argument("--buzzdev", help="Negative class",
                           type=str, default="small_guess.buzztest.jsonl")
    argparser.add_argument("--vocab", help="Vocabulary that can be features",
                           type=str, default="small_guess.vocab")
    argparser.add_argument("--passes", help="Number of passes through train",
                           type=int, default=5)
    argparser.add_argument("--batch", help="Number of items in each batch",
                           type=int, default=1)
    argparser.add_argument("--learnrate", help="Learning rate for SGD",
                           type=float, default=0.1)

    args = argparser.parse_args()

    with open(args.vocab, 'r') as infile:
        vocab = [x.strip() for x in infile]    

    train = GuessDataset(vocab)
    test = GuessDataset(vocab)

    train.initialize(args.buzztrain)
    test.initialize(args.buzzdev)

    print("Read in %i train and %i test" % (len(train), len(test)))

    # Initialize model
    logreg = SimpleLogreg(train.num_features)

    num_epochs = args.passes
    batch = args.batch
    total_samples = len(train)

    # Replace these with the correct loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(logreg.parameters(), lr=args.learnrate)

    train_loader = DataLoader(dataset=train,
                              batch_size=batch,
                              shuffle=True,
                              num_workers=0)
    dataiter = iter(train_loader)

    # Iterations
    for epoch in range(num_epochs):
        for ex, (inputs, labels) in enumerate(train_loader):
            # Run your training process
            step(epoch, ex, logreg, optimizer, criterion, inputs, labels)

    print(logreg)
    acc = logreg.evaluate(test)
    print("Accuracy: %f" % acc)
    torch.save(logreg.state_dict(), "trained_model.th")
