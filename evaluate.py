import csv
import numpy as np
import collections
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from util import get_logger
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

def link_predict(train_dataset, test_dataset, name='prediction', epoch=None, save=None):
    '''
    using macro
    Args:
        train_dataset: a dict of 
            'src_embed': a list of embedding of source node
            'dst_embed': a list of embedding of destination node
            'edge_label': 1 for valid edge, 0 for invalide edge
            for training
        test_dataset: the same structure as train_dataset but for testing
        name: hint string only for log
        epoch: an integer only for log
    '''
    logger = get_logger()
    logger.info('start link prediction %s', name)

    train_edge_num = len(train_dataset['edge_label'])
    train_edge_embed = [train_dataset['src_embed'][i] * train_dataset['dst_embed'][i] for i in range(train_edge_num)]
    logger.info('fitting models')

    model = linear_model.LogisticRegression()
    model.fit(train_edge_embed, train_dataset['edge_label'])

    test_edge_num = len(test_dataset['edge_label'])
    test_edge_embed = [test_dataset['src_embed'][i] * test_dataset['dst_embed'][i] for i in range(test_edge_num)]
    logger.info('predicting models')
    predict_label = model.predict(test_edge_embed)

    if save:
        with open('valid_' + save + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, embed in enumerate(train_edge_embed):
                line = []
                line.append(train_dataset['edge_label'][idx])
                line.extend(embed)
                writer.writerow(line)
        with open('test_' + save + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, embed in enumerate(test_edge_embed):
                line = []
                line.append(test_dataset['edge_label'][idx])
                line.append(predict_label[idx])
                line.extend(embed)
                writer.writerow(line)

    f1 = f1_score(test_dataset['edge_label'], predict_label)
    auc = roc_auc_score(test_dataset['edge_label'], model.predict_proba(test_edge_embed)[:,1])
    accuracy = accuracy_score(test_dataset['edge_label'], predict_label)
    

    scores = {'f1': f1,
              'auc_roc': auc,
              'accuracy': accuracy}
    if epoch:
        logger.info('link prediction %s epoch %d: %s',
                    name, epoch, str(scores))
    else:
        logger.info('link prediction %s : %s', name, str(scores))

def node_classify(train_dataset, test_dataset, name='classification', epoch=None, save=None):
    '''
    using micro
    Args:
        train_dataset: a dict of 
            'embed': a list of embedding of nodes
            'node_label': a list of label of nodes
            for training
        test_dataset: the same structure as train_dataset but for testing
        name: hint string only for log
        epoch: an integer only for log
    '''
    logger = get_logger()
    logger.info('start node classfication %s', name)

    label_counter = collections.Counter(test_dataset['node_label'])
    num_labels = max(train_dataset['node_label'] + test_dataset['node_label']) + 1

    logger.info('fitting models')
    model = linear_model.LogisticRegression()
    model.fit(train_dataset['embed'], train_dataset['node_label'])

    logger.info('predicting models')
    predict_label = model.predict(test_dataset['embed'])

    if save:
        with open('valid_' + save + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, embed in enumerate(train_dataset['embed']):
                line = []
                line.append(train_dataset['node_label'][idx])
                line.extend(embed)
                writer.writerow(line)
        with open('test_' + save + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            for idx, embed in enumerate(test_dataset['embed']):
                line = []
                line.append(test_dataset['node_label'][idx])
                line.append(predict_label[idx])
                line.extend(embed)
                writer.writerow(line)

    del label_counter[label_counter.most_common(1)[0][0]]#del the most common label
    sum_counter = collections.Counter()
    for label in label_counter:
        true_res = 1*(np.array(test_dataset['node_label']) == label)
        predict_res = 1*(predict_label == label)
        sum_counter.update(collections.Counter(zip(true_res, predict_res)))

    p = sum_counter[(1,1)] * 1.0 / (sum_counter[(1,1)] + sum_counter[(0,1)])
    r = sum_counter[(1,1)] * 1.0 / (sum_counter[(1,1)] + sum_counter[(1,0)])
    accuracy = accuracy_score(test_dataset['node_label'], predict_label)
    micro_f1 = 2.0/(1/p + 1/r)
    macro_f1 = f1_score(test_dataset['node_label'], predict_label, average='macro')

    scores = {'micro_f1': micro_f1,
              'macro_f1': macro_f1,
              'accuracy': accuracy}
    if epoch:
        logger.info('node classification %s epoch %d: %s',
                    name, epoch, str(scores))
    else:
        logger.info('node classification %s : %s', name, str(scores))

def noise_filter(original_attr, noised_attr, filtered_attr, epoch=None):
    '''
    evaluate the difference in original_attr part and in noise part
    Args:
        name: 'validation' or 'prediction'
    '''
    logger = get_logger()
    logger.info('start noise filter evaluation')

    filtered_original_part = original_attr * filtered_attr
    noise = noised_attr - original_attr
    filtered_noise_part = noise * filtered_attr

    num_original_part = np.sum(original_attr)
    num_noise_part = np.sum(noise)

    scores = {'original_mean': np.sum(np.abs(filtered_original_part))/num_original_part,
              'noise_mean':np.sum(np.abs(filtered_noise_part))/num_noise_part}

    if epoch:
        logger.info('noise filter evaluation epoch %d: %s',
                    epoch, str(scores))
    else:
        logger.info('noise filter evaluation: %s', str(scores))

def attr_filter(attr, attr_filters, mask, epoch=None):
    '''
    evaluate the mean of filter for different type of attr
    attr:
        a 2d array of attributes
    attr_filters: 
        a 2d array filter of attributes
    mask:
        a 1d array 0-1 mask, which is the type of attr
    '''
    logger = get_logger()
    attr0 = []
    attr1 = []
    filtered_attr_filters = attr_filters * attr
    num_valid = np.sum(attr, axis=0) #(attr, )
    mean_attr_filters = np.sum(filtered_attr_filters, axis=0) / num_valid #(attr, )
    for attr, attr_mask in zip(mean_attr_filters, mask):
        if attr_mask:
            attr1.append(attr)
        else:
            attr0.append(attr)
    scores = {'attr0_mean':np.mean(attr0),
           'attr1_mean':np.mean(attr1)}
    if epoch:
        logger.info('attr filter evaluation epoch %d: %s', epoch, str(scores))
    else:
        logger.info('attr filter evaluation %s', str(scores))
