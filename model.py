import torch
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import numpy as np
 
class StylometryLC:
    """
    Stylometry linear classifier
    Naive Bayes is used as it performs better than others

    Arguments
    ----------
    truncation : int
        number of features components to use [None => use all features]
    """
    def __init__(self, truncation=None):
        # initialize sklearn naive bayes classifier and SVD truncation
        self.clf = MultinomialNB(alpha=1.0)
        if truncation:
            self.truncated = True
            self.svd = TruncatedSVD(n_components=truncation)
        else:
            self.truncated = False

    def fit(self, x, y):
        # fit naive bayes classifier
        if self.truncated:
            self.svd.fit(x)
            x_truncated = self.svd.transform(x)
            self.clf.fit(x_truncated, y)
        else:
            self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of x given naive bayes
        if self.truncated:
            x_truncated = self.svd.transform(x)
            outputs = self.clf.predict(x_truncated)
        else:
            outputs = self.clf.predict(x)
        return outputs

    def predict_proba(self, x):
        # predict class (probability) of x given naive bayes
        if self.truncated:
            x_truncated = self.svd.transform(x)
            outputs = self.clf.predict_proba(x_truncated)
        else:
            outputs = self.clf.predict_proba(x)
        return outputs

class StylometryNN(torch.nn.Module):
    """
    Stylometry neural network
    A simple network consisting of one bidirectional GRU layer followed by two fully-connected layers

    Arguments
    ----------
    config : dictionary
        dictionary providing network parameters
    """
    def __init__(self, config):
        # initialize network layers
        super(StylometryNN,self).__init__()
        # GRU layer
        self.gru = torch.nn.GRU(config["emb_dim"], config["rnn_hid_dim"], num_layers=1, batch_first=True, bidirectional=True)
        # linear layers
        self.linear_first = torch.nn.Linear(config["rnn_hid_dim"]*2, config["dense_hid_dim"])
        self.linear_second = torch.nn.Linear(config["dense_hid_dim"], 1)
        # dropouts
        self.dropout = torch.nn.Dropout(p=0.1)
        # activations
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        # forward pass
        _, outputs = self.gru(x)
        outputs = outputs.transpose(0, 1).contiguous().view(x.shape[0], -1) 
        outputs = self.dropout(outputs)
        outputs = self.relu(self.linear_first(outputs))
        outputs = self.dropout(outputs)
        outputs = self.linear_second(outputs)
        outputs = self.sigmoid(outputs)
        return outputs
