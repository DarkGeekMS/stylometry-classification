import torch
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import numpy as np
 
class StylometryLC:
    def __init__(self, truncation=None):
        self.clf = MultinomialNB(alpha=1.0)
        if truncation:
            self.truncated = True
            self.svd = TruncatedSVD(n_components=truncation)
        else:
            self.truncated = False

    def fit(self, x, y):
        if self.truncated:
            self.svd.fit(x)
            x_truncated = self.svd.transform(x)
            self.clf.fit(x_truncated, y)
        else:
            self.clf.fit(x, y)

    def predict(self, x):
        if self.truncated:
            x_truncated = self.svd.transform(x)
            outputs = self.clf.predict(x_truncated)
        else:
            outputs = self.clf.predict(x)
        return outputs

class StylometryNN(torch.nn.Module):
    def __init__(self, config):
        super(StylometryNN,self).__init__()
        self.lstm = torch.nn.LSTM(config["emb_dim"], config["rnn_hid_dim"], num_layers=1, batch_first=True, dropout=0.3, bidirectional=True)
        self.linear_first = torch.nn.Linear(config["rnn_hid_dim"]*2, config["dense_hid_dim_1"])
        self.dropout_first = torch.nn.Dropout(p=0.8)
        self.linear_second = torch.nn.Linear(config["dense_hid_dim_1"], config["dense_hid_dim_2"])
        self.dropout_second = torch.nn.Dropout(p=0.8)
        self.linear_third = torch.nn.Linear(config["dense_hid_dim_2"], 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        outputs, _ = self.lstm(x)
        outputs = outputs.permute(1, 0, 2)
        outputs = self.relu(self.linear_first(outputs[-1]))
        outputs = self.dropout_first(outputs)
        outputs = self.relu(self.linear_second(outputs))
        outputs = self.dropout_second(outputs)
        outputs = self.relu(self.linear_third(outputs))
        outputs = self.sigmoid(outputs)
        return outputs
