import argparse
import pickle
import os
import logging
from util import get_logger
from model import Model
import dataset
import prepare

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

    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def parse_args():
    '''
    parse arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--predict', action='store_true')

    parser.add_argument(
        '--dataset', choices=['facebook', 'cora', 'amazon', 'citeseer', 'pubmed'], default='facebook')
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


def main():
    '''
    main
    '''
    set_env()
    args = parse_args()
    logger = get_logger()
    logger.info(args)

    # build graph
    if args.prepare:
        logger.info('start preparing')
        data_dir = args.data_dir
        if args.dataset == 'facebook':
            prepare.prepare_facebook(
                data_dir + '/facebook', data_dir, data_dir + '/prepare')
            graph = prepare.build_graph_facebook(
                data_dir + '/prepare', noise=args.noise)
        elif args.dataset == 'cora':
            prepare.prepare_cora(data_dir + '/cora', data_dir + '/cora')
            graph = prepare.build_graph_cora(
                data_dir + '/cora', noise=args.noise)
        elif args.dataset == 'amazon':
            prepare.prepare_amazon(data_dir, data_dir)
            graph = prepare.build_graph_amazon(data_dir, noise=args.noise)
        elif args.dataset == 'citeseer':
            prepare.prepare_citeseer(data_dir + '/citeseer', data_dir + '/citeseer')
            graph = prepare.build_graph_citeseer(
                data_dir + '/citeseer', noise=args.noise)
        elif args.dataset == 'pubmed':
            graph = prepare.build_graph_pubmed(
                data_dir + '/data', noise=args.noise)

        dataset.prepare_dataset(
            graph, args.train_batch_size, args.eval_batch_size, 
            task=args.task, train_percent=0.8, save_path='./prepare')

    if args.train:
        logger.info('start training')
        with open('./prepare/graph.pk', 'rb') as f:
            graph = pickle.load(f)
        model = Model(graph, args)
        # train
        model.train()
        model.restore('model')
        model.predict()

    if args.resume:
        logger.info('resume from %d', args.epoch_base)
        with open('./prepare/graph.pk', 'rb') as f:
            graph = pickle.load(f)
        model = Model(graph, args)
        model.restore()
        model.train()
        model.restore('model')
        model.predict()
    
    if args.predict:
        logger.info('predicting')
        with open('./prepare/graph.pk', 'rb') as f:
            graph = pickle.load(f)
        model = Model(graph, args)
        model.restore('model')
        model.predict()

if __name__ == '__main__':
    main()
