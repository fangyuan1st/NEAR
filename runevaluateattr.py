import logging
import argparse
import os
import pickle
import re
import numpy as np
from model import Model
from util import get_logger
import evaluate


def set_env():
    '''
    set logger and environment
    '''
    logger = get_logger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler('./log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s -%(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

def parse_args():
    '''
    parse arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')

    parser.add_argument(
        '--dataset', choices=['facebook', 'cora'], default='facebook')
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--noise', type=float, default=None,
                        help='noise rate for attribute 0->1 when prepare')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--neg_sample_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--epoch_base', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)

    return parser.parse_args()

def main():
    set_env()
    args = parse_args()
    logger = get_logger()

    with open(os.path.join(args.data_dir, 'graph.pk'), 'rb') as f:
        graph = pickle.load(f)
    model = Model(graph, args)
    model.restore()

    logger.info('handling mask')
    mask = []
    pattern = re.compile(r'birthday;|first_name;|middle_name;|name;|gender;|work;end_date;|work;start_date;')
    with open(os.path.join(os.path.join(args.data_dir, 'prepare'), 'combined_featnames.txt'), 'r') as fin:
        for line in fin:
            line = line.split()
            idx = line[0]
            name = line[1]
            if pattern.match(name):
                mask.append(1)
            else:
                mask.append(0)
    
    logger.info('handling attr')
    attr_filter_matrix = model.sess.run(model.attr_filter_matrix)

    evaluate.attr_filter(np.array(graph.attribute), attr_filter_matrix, mask)

if __name__ == '__main__':
    main()
