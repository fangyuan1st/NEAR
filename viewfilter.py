
import logging
import argparse
import os
import pickle
import re
import numpy as np
# import matplotlib.pyplot as plt
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
    parser.add_argument(
        '--task', choices=['link_predict', 'node_classify', 'all', 'none'], default='link_predict')
    parser.add_argument('--model_dir', default='.')
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--noise', type=float, default=None,
                        help='noise rate for attribute 0->1 when prepare')
    parser.add_argument('--embed_size', type=int, default=64)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--neg_sample_num', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    parser.add_argument('--epoch_base', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)

    return parser.parse_args()

def filter_hist(original_attr, noised_attr, filtered_attr):
    '''
    evaluate the difference in original_attr part and in noise part
    Args:
        name: 'validation' or 'prediction'
    '''
    logger = get_logger()
    logger.info('start noise filter evaluation')

    original_mask = original_attr
    noise_mask = noised_attr - original_attr

    original_part = []
    noise_part = []
    for x, row in enumerate(filtered_attr):
        original_part.extend([entry for y, entry in enumerate(row) if original_mask[x][y]])
        noise_part.extend([entry for y, entry in enumerate(row) if noise_mask[x][y]])

    original_weights = np.ones_like(original_part)/float(len(original_part))
    noise_weights = np.ones_like(noise_part)/float(len(noise_part))

    with open('original_part.txt', 'w') as fout:
        for entry in original_part:
            fout.write(str(entry) + '\n')
    
    with open('noise_weights.txt', 'w') as fout:
        for entry in noise_part:
            fout.write(str(entry) + '\n')

    
    # original_percent, original_bins, _ = plt.hist(original_part, bins=50, weights=original_weights)
    # noise_percent, noise_bins, _ = plt.hist(noise_part, bins=50, weights=noise_weights)
    # original_bins = original_bins[:-1]
    # noise_bins = noise_bins[:-1]

    # print(np.mean(original_part))
    # print(np.mean(noise_part))

    # plt.close('all')
    # plt.figure()
    # plt.plot(original_bins, original_percent, label='origin')
    # plt.plot(noise_bins, noise_percent, label='noise')
    # plt.legend()
    # plt.show()

def main():
    set_env()
    args = parse_args()
    logger = get_logger()

    logger.info('reading graph')
    with open('./prepare/graph.pk', 'rb') as f:
        graph = pickle.load(f)

    logger.info('building model')
    model = Model(graph, args)
    model.restore('model')

    logger.info('preparing attributes')
    origin_attribute = np.array(graph.origin_attribute)
    noised_attribute = np.array(graph.attribute)
    attr_filter = model.sess.run(model.attr_filter_matrix)
    denoised_attribute = noised_attribute * attr_filter

    filter_hist(origin_attribute, noised_attribute, denoised_attribute)

if __name__ == '__main__':
    main()
