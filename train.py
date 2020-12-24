from text_dataset import TextDataset
from model import StylometryLC, StylometryNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import log_loss, accuracy_score

import numpy as np
import argparse
from tqdm import tqdm
import pickle
import json
import sys

def nn_train(config):
    # train neural network classifier
    # select device (CPU | GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create and build dataset
    dataset = TextDataset(config['txt_list'], norm=config['norm'], vectorizer=config['vectorizer'], w2v_path=config['w2v_path'])
    xtrain, xvalid, ytrain, yvalid = dataset.build_dataset()
    # define network configuration
    network_config = {
        'emb_dim'         : config['emb_dim'],
        'rnn_hid_dim'     : config['rnn_hid_dim'],
        'dense_hid_dim' : config['dense_hid_dim'],
    }
    # define model
    model = StylometryNN(network_config)
    model.to(device)
    model.train()
    # define BCE loss
    criterion = nn.BCELoss()
    # define optimizer
    optimizer = Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    train_steps = int(len(xtrain) / config['batch_size'])
    valid_steps = int(len(xvalid) / config['batch_size'])
    best_accuracy = 0.5
    # training loop
    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        # loop over all training dataset samples
        for step in tqdm(range(train_steps)):
            # get x and y batches
            x_batch = xtrain[step*config['batch_size']:(step+1)*config['batch_size']]
            x_batch = np.stack(x_batch, axis=0)
            x_batch = torch.from_numpy(x_batch).float().to(device)
            y_batch = ytrain[step*config['batch_size']:(step+1)*config['batch_size']]
            y_batch = np.stack(y_batch, axis=0)
            y_batch = torch.from_numpy(y_batch).float().to(device)
            # model forward pass
            y_out = model(x_batch)
            y_out = torch.squeeze(y_out, dim=1)
            # calculate loss
            loss = criterion(y_out, y_batch)
            total_loss += loss.item()
            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{int(epoch + 1)}/{int(config["num_epochs"])}], Total Epoch Loss: {total_loss/train_steps}')
        model.eval()
        accuracies = list()
        losses = list()
        # loop over all validation dataset samples
        for step in tqdm(range(valid_steps)):
            # get x and y batches
            x_batch = xvalid[step*config['batch_size']:(step+1)*config['batch_size']]
            x_batch = np.stack(x_batch, axis=0)
            x_batch = torch.from_numpy(x_batch).float().to(device)
            y_batch = yvalid[step*config['batch_size']:(step+1)*config['batch_size']]
            y_batch = np.stack(y_batch, axis=0)
            # model forward pass
            y_out = model(x_batch)
            y_out = torch.squeeze(y_out, dim=1).cpu().detach().numpy()
            # calculate loss and accuracy
            y_out_labels = y_out > 0.5
            accuracies.append(accuracy_score(y_batch, y_out_labels))
            losses.append(F.binary_cross_entropy(torch.from_numpy(y_out).float(), torch.from_numpy(y_batch).float()))
        # print results
        print (f'Validation accuracy: {(sum(accuracies)/len(accuracies))*100}%')
        print (f'Validation logloss: {sum(losses)/len(losses)}')
        # save model (based on best validation accuracy)
        if sum(accuracies)/len(accuracies) > best_accuracy:
            torch.save(model, 'models/deep_model.pt')
        model.train()

def lc_train(config):
    # train linear classifier (Naive Bayes)
    # create and build dataset
    dataset = TextDataset(config['txt_list'], norm=config['norm'], vectorizer=config['vectorizer'])
    xtrain, xvalid, ytrain, yvalid = dataset.build_dataset()
    # define model
    model = StylometryLC(truncation=config['truncation'])
    # fit model
    model.fit(xtrain, ytrain)
    # infer on validation data
    predictions = model.predict(xvalid)
    predictions_proba = model.predict_proba(xvalid)
    # dump model pickle
    pickle.dump(model, open('models/nb_model.sav', 'wb'))
    # print results
    print (f'accuracy: {accuracy_score(yvalid, predictions)*100}%')
    print (f'logloss: {log_loss(yvalid, predictions_proba)}')

if __name__ == "__main__":
    # argument parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    subparsers = argparser.add_subparsers(help='Sub-commands', dest='command')
    # neural network subparser
    nn_subparser = subparsers.add_parser('train-nn', help='Train neural network')
    nn_subparser.add_argument('--config', help='path to network config file', dest='config', default='configs/nn_config.json')
    # linear classifier subparser
    lc_subparser = subparsers.add_parser('train-lc', help='Train linear classifier')
    lc_subparser.add_argument('--config', help='path to model config file', dest='config', default='configs/lc_config.json')

    args = argparser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')
    config_file = kwargs.pop('config')

    # read config JSON
    with open(config_file, 'r') as f:
        config = json.load(f)

    # call training function
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)
    elif subcmd == 'train-nn':
        nn_train(config)
    elif subcmd == 'train-lc':
        lc_train(config)
