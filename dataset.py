import pickle
import os
import random
import numpy as np
from util import get_logger
from graph import Graph


def jaccard_similarity(graph, i, j):
    '''
    jaccard_similarity
    '''
    neighbor1 = set(graph.graph[i])
    neighbor2 = set(graph.graph[j])
    if len(neighbor1 | neighbor2) == 0:
        return 0
    return len(neighbor1 & neighbor2) * 1.0 / len(neighbor1 | neighbor2)


def prepare_train(graph, train_edges, save_name, batch_size):
    '''
    shuffled
    '''
    np.random.seed(0)
    np.random.shuffle(train_edges)

    dataset = []
    batch = {'center_id': [], 'neighbor_id': [],
             'center_attr': [], 'neighbor_attr': [], 'Sij': []}
    for idx, edge in enumerate(train_edges, 1):
        batch['center_id'].append(edge[0])
        batch['neighbor_id'].append(edge[1])
        batch['center_attr'].append(graph.attribute[edge[0]])
        batch['neighbor_attr'].append(graph.attribute[edge[1]])
        batch['Sij'].append(jaccard_similarity(graph, edge[0], edge[1]))
        if idx % batch_size == 0:
            dataset.append(batch)
            batch = {'center_id': [], 'neighbor_id': [],
                     'center_attr': [], 'neighbor_attr': [], 'Sij': []}

    if len(batch['center_id']):
        dataset.append(batch)

    with open(save_name, 'wb') as f:
        pickle.dump(dataset, f)


def sample_neg_edges(graph, valid_edges, evaluate_edges=None):
    '''
    sample negtative edges both 2hop and random
    Args:
        evaluate_edges: used when generate predict set, sampling edges not contained in evaluate_edges
    '''
    np.random.seed(0)
    random.seed(0)
    logger = get_logger()
    logger.info('sampling...')
    valid_edges = [(edge[0], edge[1], 1) for edge in valid_edges]
    neg_hop2_edges = []
    node_num = graph.node_num()
    for node in range(node_num):
        for neighbor in graph.graph[node]:
            hop2 = [(node, hop2, 0)
                    for hop2 in graph.graph[neighbor] if hop2 not in graph.graph[node]]
            neg_hop2_edges += hop2
    logger.info('neg_hop2_edges generate finish')

    if evaluate_edges:
        random_neg_num = len(valid_edges) - int(len(valid_edges) / 2)
        neg_hop2_edges = list(set(neg_hop2_edges) - set(evaluate_edges))
        for i in range(int(len(valid_edges) / 2)):
            valid_edges.append(random.choice(neg_hop2_edges))
        logger.info('neg_hop2_edges sampling finish')

        while random_neg_num>0:
            temp = []
            for i in range(10000):
                node1 = random.randint(0, node_num - 1)
                node2 = random.randint(0, node_num - 1)
                if node2 not in graph.graph[node1]:
                    temp.append((node1, node2, 0))
            temp = list(set(temp) - set(evaluate_edges))
            if len(temp) > random_neg_num:
                temp = temp[:random_neg_num]
            random_neg_num -= len(temp)
            valid_edges += temp

    else:
        random_neg_num = len(valid_edges) - int(len(valid_edges) / 2)
        for i in range(int(len(valid_edges) / 2)):
            valid_edges.append(random.choice(neg_hop2_edges))
        logger.info('neg_hop2_edges sampling finish')

        while random_neg_num > 0:
            node1 = random.randint(0, node_num - 1)
            node2 = random.randint(0, node_num - 1)
            if node2 not in graph.graph[node1]:
                valid_edges.append((node1, node2, 0))
                random_neg_num -= 1

    logger.info("valid_set: total: %d, true: %d",
                len(valid_edges), sum([edge[2] for edge in valid_edges]))

    np.random.shuffle(valid_edges)

    return valid_edges


def prepare_link_predict_eval(graph, valid_edges, save_name, batch_size, evaluate_edges=None):
    '''
    dataset for predict
    Args:
        evaluate_edges: used when generate predict set, sampling edges not contained in evaluate_edges
    '''
    edges = sample_neg_edges(graph, valid_edges, evaluate_edges)
    dataset = []

    batch = {'node_ids': [], 'node_attrs': [], 'edges': [], 'node2id': {}}
    for idx, edge in enumerate(edges, 1):
        if edge[0] not in batch['node_ids']:
            batch['node2id'][edge[0]] = len(batch['node_ids'])
            batch['node_ids'].append(edge[0])
            batch['node_attrs'].append(graph.attribute[edge[0]])

        if edge[1] not in batch['node_ids']:
            batch['node2id'][edge[1]] = len(batch['node_ids'])
            batch['node_ids'].append(edge[1])
            batch['node_attrs'].append(graph.attribute[edge[1]])

        batch['edges'].append(
            (batch['node2id'][edge[0]], batch['node2id'][edge[1]], edge[2]))

        if idx % batch_size == 0:
            del batch['node2id']
            dataset.append(batch)
            batch = {'node_ids': [], 'node_attrs': [],
                     'edges': [], 'node2id': {}}

    if len(batch['node_ids']):
        del batch['node2id']
        dataset.append(batch)

    with open(save_name, 'wb') as fout:
        pickle.dump(dataset, fout)

    return edges


def prepare_node_classify_eval(graph, nodes, save_name, batch_size):
    '''
    dataset for predict
    '''
    dataset = []
    batch = {'node_ids':[], 'node_attrs':[], 'node_labels':[]}
    for idx, node_id in enumerate(nodes, 1):
        batch['node_ids'].append(node_id)
        batch['node_attrs'].append(graph.attribute[node_id])
        batch['node_labels'].append(graph.label[node_id])

        if idx % batch_size == 0:
            dataset.append(batch)
            batch = {'node_ids':[], 'node_attrs':[], 'node_labels':[]}
    
    if len(batch['node_ids']):
        dataset.append(batch)
    
    with open(save_name, 'wb') as fout:
        pickle.dump(dataset, fout)


def prepare_dataset(graph, train_batch_size, eval_batch_size, task='link_predict', train_percent=0.5, save_path='.'):
    '''
    generate train, valid, test dataset, and save datasets and train graph as save_path
    task can be 'link_predict' or 'node_classify'
    '''
    np.random.seed(0)
    logger = get_logger()
    np.random.shuffle(graph.edges)
    edge_num = len(graph.edges)
    # directed or indirected
    train_edges = graph.edges[:int(edge_num * train_percent)]
    valid_percent = (1-train_percent)*1.0/2
    valid_edges = graph.edges[int(edge_num * train_percent):
                              int(edge_num * (train_percent + valid_percent))]
    test_edges = graph.edges[int(edge_num * (train_percent + valid_percent)):]

    logger.info('building train_graph')
    train_graph = Graph(graph.is_directed())
    for edge in train_edges:
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        if node0 not in train_graph.node2id:
            train_graph.add_node(node0, 
                                 attribute=graph.attribute[edge[0]],
                                 origin_attribute=graph.origin_attribute[edge[0]],
                                 label=graph.label[edge[0]])
        if node1 not in train_graph.node2id:
            train_graph.add_node(node1, 
                                 attribute=graph.attribute[edge[1]],
                                 origin_attribute=graph.origin_attribute[edge[1]],
                                 label=graph.label[edge[1]])
        train_graph.add_edge(node0, node1)
    
    logger.info('building reprojection')
    sorted_ids = train_graph.sort_by_degree(
        range(train_graph.node_num()), reverse=True)
    projection = [None for i in range(train_graph.node_num())]
    for idx, nodeid in enumerate(sorted_ids):
        projection[nodeid] = idx
    train_graph.reprojection(projection)

    logger.info('filter edges')
    # only nodes in train dataset
    valid_edges = [x for x in valid_edges
                   if (graph.id2node[x[0]] in train_graph.node2id) and
                   (graph.id2node[x[1]] in train_graph.node2id)]
    test_edges = [x for x in test_edges
                  if (graph.id2node[x[0]] in train_graph.node2id) and
                  (graph.id2node[x[1]] in train_graph.node2id)]

    for edge in valid_edges:
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        train_graph.add_edge(node0, node1)
    for edge in test_edges:
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        train_graph.add_edge(node0, node1)

    logger.info('update node_ids')
    #update new node_ids on edges
    for idx, edge in enumerate(train_edges):
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        train_edges[idx] = (train_graph.node2id[node0], train_graph.node2id[node1])
    
    for idx, edge in enumerate(valid_edges):
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        valid_edges[idx] = (train_graph.node2id[node0], train_graph.node2id[node1])
    
    for idx, edge in enumerate(test_edges):
        node0 = graph.id2node[edge[0]]
        node1 = graph.id2node[edge[1]]
        test_edges[idx] = (train_graph.node2id[node0], train_graph.node2id[node1])

    #if is undirected graph
    if not train_graph.is_directed():
        for i in range(len(train_edges)):
            train_edges.append((train_edges[i][1], train_edges[i][0]))
        for i in range(len(valid_edges)):
            valid_edges.append((valid_edges[i][1], valid_edges[i][0]))
        for i in range(len(test_edges)):
            test_edges.append((test_edges[i][1], test_edges[i][0]))

    logger.info('preparing train dataset')
    prepare_train(train_graph, train_edges, os.path.join(
        save_path, 'train_dataset.pk'), train_batch_size)
    logger.info('finish train dataset')

    logger.info('preparing eval dataset')
    prepare_train(train_graph, valid_edges, os.path.join(
        save_path, 'eval_dataset.pk'), eval_batch_size)
    logger.info('finish eval dataset')

    logger.info('add valid_edges test_edges')
    #add valid edges and test edges to graph
    for edge in valid_edges:
        train_graph.add_edge(edge[0], edge[1])
    for edge in test_edges:
        train_graph.add_edge(edge[0], edge[1])

    if task=='link_predict' or task == 'all':
        logger.info('preparing valid dataset')
        valid_edges = prepare_link_predict_eval(train_graph, valid_edges, os.path.join(
            save_path, 'valid_dataset.pk'), eval_batch_size, None)
        logger.info('finish valid dataset')

        logger.info('preparing test dataset')
        prepare_link_predict_eval(train_graph, test_edges, os.path.join(
            save_path, 'test_dataset.pk'), eval_batch_size, valid_edges)
        logger.info('finish test dataset')
    if task=='node_classify' or task == 'all':
        np.random.seed(0)
        train_nodes = []
        predict_nodes = []

        stratified_nodes = {}
        for node_id in range(train_graph.node_num()):
            label = train_graph.label[node_id]
            if label not in stratified_nodes:
                stratified_nodes[label] = []
            stratified_nodes[label].append(node_id)
        
        for label in stratified_nodes:
            nodes = stratified_nodes[label]
            np.random.shuffle(nodes)
            train_nodes.extend(nodes[:int(0.8 * len(nodes))])
            predict_nodes.extend(nodes[int(0.8 * len(nodes)):])

        np.random.shuffle(train_nodes)
        np.random.shuffle(predict_nodes)
            
        logger.info('node classification %d: %d', len(train_nodes), len(predict_nodes))

        logger.info('preparing valid dataset')
        prepare_node_classify_eval(train_graph, train_nodes, os.path.join(
            save_path, 'valid_dataset_node.pk'), eval_batch_size)
        logger.info('finish valid dataset')

        logger.info('prepare test dataset')
        prepare_node_classify_eval(train_graph, predict_nodes, os.path.join(
            save_path, 'test_dataset_node.pk'), eval_batch_size)
        logger.info('finish test dataset')

    with open(os.path.join(save_path, 'graph.pk'), 'wb') as fout:
        pickle.dump(train_graph, fout)
    logger.info('write graph.pk to %s', save_path)


def gen_batches(save_name):
    '''
    load batches from existing dataset
    Return:
        yield one batch at one time
    '''
    with open(save_name, 'rb') as fin:
        dataset = pickle.load(fin)

    for batch in dataset:
        yield batch
