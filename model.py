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
        self.gru = torch.nn.GRU(config["emb_dim"], config["rnn_hid_dim"], num_layers=1, batch_first=True, bidirectional=True)
        self.linear_first = torch.nn.Linear(config["rnn_hid_dim"]*2, config["dense_hid_dim"])
        self.linear_second = torch.nn.Linear(config["dense_hid_dim"], 1)
        self.dropout = torch.nn.Dropout(p=0.1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x):
        _, outputs = self.gru(x)
        outputs = outputs.transpose(0, 1).contiguous().view(x.shape[0], -1) 
        outputs = self.dropout(outputs)
        outputs = self.relu(self.linear_first(outputs))
        outputs = self.linear_second(outputs)
        outputs = self.sigmoid(outputs)
        return outputs
