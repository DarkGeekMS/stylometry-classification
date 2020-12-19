from text_dataset import TextDataset
from model import StylometryLC, StylometryNN

from sklearn.metrics import log_loss, accuracy_score

import argparse
import json
import sys

def nn_train(config):
    pass

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
