from text_dataset import TextDataset

import torch
import torch.nn as nn
from sklearn.metrics import log_loss, accuracy_score

import numpy as np
import argparse
from tqdm import tqdm
import pickle
import sys

def nn_train(author1, author2, model, w2v_path):
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TextDataset([author1, author2], norm=None, vectorizer='embed', w2v_path=w2v_path)
    X, _, Y, _ = dataset.build_dataset(eval=True)
    model = torch.load(model)
    model.to(device)
    model.eval()
    valid_steps = int(len(X) / batch_size)
    accuracies = list()
    losses = list()
    for step in tqdm(range(valid_steps)):
        x_batch = X[step*batch_size:(step+1)*batch_size]
        x_batch = np.stack(x_batch, axis=0)
        x_batch = torch.from_numpy(x_batch).float().to(device)
        y_batch = Y[step*batch_size:(step+1)*batch_size]
        y_batch = np.stack(y_batch, axis=0)
        y_out = model(x_batch)
        y_out = torch.squeeze(y_out, dim=1).cpu().detach().numpy()
        y_out = y_out > 0.5
        accuracies.append(accuracy_score(y_batch, y_out))
        losses.append(log_loss(y_batch, y_out))
    print (f'accuracy: {(sum(accuracies)/len(accuracies))*100}%')
    print (f'logloss: {sum(losses)/len(losses)}')

def lc_train(author1, author2, model):
    dataset = TextDataset([author1, author2], norm=None, vectorizer='tfidf')
    X, _, Y, _ = dataset.build_dataset(eval=True)
    model = pickle.load(open(model, 'rb'))
    predictions = model.predict(X)
    print (f'accuracy: {accuracy_score(Y, predictions)*100}%')
    print (f'logloss: {log_loss(Y, predictions)}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    subparsers = argparser.add_subparsers(help='Sub-commands', dest='command')
    nn_subparser = subparsers.add_parser('eval-nn', help='Train neural network')
    nn_subparser.add_argument('--author1', help='path to first author writings', dest='author1', default='data/EAP_train.txt')
    nn_subparser.add_argument('--author2', help='path to second author writings', dest='author2', default='data/MWS_train.txt')
    nn_subparser.add_argument('--model', help='path to model file', dest='model', default='models/deep_model.pt')
    nn_subparser.add_argument('--w2v_path', help='path to word2vec file', dest='w2v_path', default='w2v_models/glove.840B.300d.txt')

    lc_subparser = subparsers.add_parser('eval-lc', help='Train linear classifier')
    lc_subparser.add_argument('--author1', help='path to first author writings', dest='author1', default='data/EAP_train.txt')
    lc_subparser.add_argument('--author2', help='path to second author writings', dest='author2', default='data/MWS_train.txt')
    lc_subparser.add_argument('--model', help='path to model file', dest='model', default='models/nb_model.sav')

    args = argparser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')
    author1 = kwargs.pop('author1')
    author2 = kwargs.pop('author2')
    model = kwargs.pop('model')
    w2v_path = kwargs.pop('w2v_path')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)
    elif subcmd == 'eval-nn':
        nn_train(author1, author2, model, w2v_path)
    elif subcmd == 'eval-lc':
        lc_train(author1, author2, model)
