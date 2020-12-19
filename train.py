from text_dataset import TextDataset
from model import StylometryLC, StylometryNN

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import log_loss, accuracy_score

import numpy as np
import argparse
from tqdm import tqdm
import json
import sys

def nn_train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TextDataset(config['txt_list'], norm=config['norm'], vectorizer=config['vectorizer'], w2v_path=config['w2v_path'])
    xtrain, xvalid, ytrain, yvalid = dataset.build_dataset()
    network_config = {
        'emb_dim'         : config['emb_dim'],
        'rnn_hid_dim'     : config['rnn_hid_dim'],
        'dense_hid_dim_1' : config['dense_hid_dim_1'],
        'dense_hid_dim_2' : config['dense_hid_dim_2']
    }
    model = StylometryNN(network_config)
    model.to(device)
    model.train()
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=config['initial_lr'], weight_decay=config['weight_decay'])
    train_steps = int(len(xtrain) / config['batch_size'])
    valid_steps = int(len(xvalid) / config['batch_size'])
    for epoch in range(config['num_epochs']):
        total_loss = 0.0
        for step in tqdm(range(train_steps)):
            x_batch = xtrain[step*config['batch_size']:(step+1)*config['batch_size']]
            x_batch = np.stack(x_batch, axis=0)
            x_batch = torch.from_numpy(x_batch).float().to(device)
            y_batch = ytrain[step*config['batch_size']:(step+1)*config['batch_size']]
            y_batch = np.stack(y_batch, axis=0)
            y_batch = torch.from_numpy(y_batch).float().to(device)
            y_out = model(x_batch)
            y_out = torch.squeeze(y_out, dim=1)
            loss = criterion(y_out, y_batch)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch [{}/{}], Total Epoch Loss: total loss: {:.4f}'.format(epoch + 1, config['num_epochs'], total_loss/train_steps))
    model.eval()
    predictions = list()
    for step in tqdm(range(valid_steps)):
        x_batch = xvalid[step*config['batch_size']:(step+1)*config['batch_size']]
        x_batch = np.stack(x_batch, axis=0)
        x_batch = torch.from_numpy(x_batch).float().to(device)
        y_out = model(x_batch)
        y_out = y_out.to('cpu')
        predictions.append(y_out)
    predictions = np.concatenate(predictions, axis=0)
    print (f'accuracy: {accuracy_score(yvalid, predictions)*100}%')
    print (f'logloss: {log_loss(yvalid, predictions)}')

def lc_train(config):
    dataset = TextDataset(config['txt_list'], norm=config['norm'], vectorizer=config['vectorizer'])
    xtrain, xvalid, ytrain, yvalid = dataset.build_dataset()
    model = StylometryLC(truncation=config['truncation'])
    model.fit(xtrain, ytrain)
    predictions = model.predict(xvalid)
    print (f'accuracy: {accuracy_score(yvalid, predictions)*100}%')
    print (f'logloss: {log_loss(yvalid, predictions)}')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    subparsers = argparser.add_subparsers(help='Sub-commands', dest='command')
    nn_subparser = subparsers.add_parser('train-nn', help='Train neural network')
    nn_subparser.add_argument('--config', help='path to network config file', dest='config', default='configs/nn_config.json')

    lc_subparser = subparsers.add_parser('train-lc', help='Train linear classifier')
    lc_subparser.add_argument('--config', help='path to model config file', dest='config', default='configs/lc_config.json')

    args = argparser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')
    config_file = kwargs.pop('config')

    with open(config_file, 'r') as f:
        config = json.load(f)

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)
    elif subcmd == 'train-nn':
        nn_train(config)
    elif subcmd == 'train-lc':
        lc_train(config)
