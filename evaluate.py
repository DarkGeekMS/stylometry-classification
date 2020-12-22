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
    # evaluate neural network classifier
    # define batch size
    batch_size = 128
    # select device (CPU | GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create and build evaluation dataset
    dataset = TextDataset([author1, author2], norm=None, vectorizer='embed', w2v_path=w2v_path)
    X, _, Y, _ = dataset.build_dataset(eval=True)
    # load model from file
    model = torch.load(model)
    model.to(device)
    model.eval()
    valid_steps = int(len(X) / batch_size)
    predictions = list()
    # loop over all evaluation dataset
    for step in tqdm(range(valid_steps)):
        # get x and y batches
        x_batch = X[step*batch_size:(step+1)*batch_size]
        x_batch = np.stack(x_batch, axis=0)
        x_batch = torch.from_numpy(x_batch).float().to(device)
        y_batch = Y[step*batch_size:(step+1)*batch_size]
        y_batch = np.stack(y_batch, axis=0)
        # model forward pass
        y_out = model(x_batch)
        y_out = torch.squeeze(y_out, dim=1).cpu().detach().numpy()
        # save predictions
        y_out = y_out > 0.5
        predictions.append(y_out)
    # perform inference on the remaining samples
    x_batch = X[valid_steps*batch_size:]
    x_batch = np.stack(x_batch, axis=0)
    x_batch = torch.from_numpy(x_batch).float().to(device)
    y_batch = Y[valid_steps*batch_size:]
    y_batch = np.stack(y_batch, axis=0)
    y_out = model(x_batch)
    y_out = torch.squeeze(y_out, dim=1).cpu().detach().numpy()
    y_out = y_out > 0.5
    predictions.append(y_out)
    # concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    # print results
    print (f'accuracy: {accuracy_score(Y, predictions)*100}%')
    print (f'logloss: {log_loss(Y, predictions)}')

def lc_train(author1, author2, model):
    # evaluate linear classifier model
    # create and build evaluation dataset
    dataset = TextDataset([author1, author2], norm=None, vectorizer='tfidf')
    X, _, Y, _ = dataset.build_dataset(eval=True)
    # load model from pickle
    model = pickle.load(open(model, 'rb'))
    # predict dataset labels
    predictions = model.predict(X)
    # print results
    print (f'accuracy: {accuracy_score(Y, predictions)*100}%')
    print (f'logloss: {log_loss(Y, predictions)}')

if __name__ == "__main__":
    # argument parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    subparsers = argparser.add_subparsers(help='Sub-commands', dest='command')
    # neural network subparser
    nn_subparser = subparsers.add_parser('eval-nn', help='Train neural network')
    nn_subparser.add_argument('--author1', help='path to first author writings', dest='author1', default='data/HPL_test.txt')
    nn_subparser.add_argument('--author2', help='path to second author writings', dest='author2', default='data/MWS_test.txt')
    nn_subparser.add_argument('--model', help='path to model file', dest='model', default='models/deep_model.pt')
    nn_subparser.add_argument('--w2v_path', help='path to word2vec file', dest='w2v_path', default='w2v_models/glove.840B.300d.txt')
    # linear classifier subparser
    lc_subparser = subparsers.add_parser('eval-lc', help='Train linear classifier')
    lc_subparser.add_argument('--author1', help='path to first author writings', dest='author1', default='data/HPL_test.txt')
    lc_subparser.add_argument('--author2', help='path to second author writings', dest='author2', default='data/MWS_test.txt')
    lc_subparser.add_argument('--model', help='path to model file', dest='model', default='models/nb_model.sav')

    args = argparser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')
    author1 = kwargs.pop('author1')
    author2 = kwargs.pop('author2')
    model = kwargs.pop('model')

    # call evaluation function
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)
    elif subcmd == 'eval-nn':
        w2v_path = kwargs.pop('w2v_path')
        nn_train(author1, author2, model, w2v_path)
    elif subcmd == 'eval-lc':
        lc_train(author1, author2, model)
