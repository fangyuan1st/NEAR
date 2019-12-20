import os
import shutil
import pickle
from graph import Graph
from util import get_logger


def combine_features(path, save_path, network_names):
    '''
    combine features in networks together, write to file
    Return:
        id2attr, attr2id
    '''
    logger = get_logger()
    attr_names = set()
    for network_name in sorted(network_names):
        logger.info('reading featnames in %s', network_name)
        with open(os.path.join(path, network_name + '.featnames')) as f:
            for line in f.readlines():
                line = line.strip().split()
                name = (' '.join(line[1:-1]), int(line[-1]))
                if name not in attr_names:
                    attr_names.add(name)

    attr_names = [' '.join([name[0], str(name[1])])
                  for name in sorted(attr_names)]

    # build id2attr and attr2id
    id2attr = attr_names
    attr2id = dict()
    for idx, name in enumerate(id2attr):
        attr2id[name] = idx

    # save combined features
    with open(os.path.join(save_path, 'combined_featnames.txt'), 'w') as fout:
        for idx, name in enumerate(id2attr):
            fout.write(str(idx) + ' ' + name + '\n')

    return id2attr, attr2id


def update_features(path, save_path, id2attr, attr2id, network_names):
    '''
    update features in each network, combine egofeat and feature
    write new egofeat and feat to prepare folder
    '''
    logger = get_logger()
    for network_name in sorted(network_names):
        logger.info('update features in %s', network_name)
        # feature id to new feature id table
        id2newid = list()
        with open(os.path.join(path, network_name + '.featnames')) as fin:
            for line in fin.readlines():
                name = ' '.join(line.strip().split()[1:])
                if name not in attr2id:
                    logger.error('featname not found in attr2id %s', name)
                    return None
                else:
                    id2newid.append(attr2id[name])

        # convert feature for egofeat and feat
        with open(os.path.join(save_path, network_name + '.egofeat'), 'w') as fout:
            with open(os.path.join(path, network_name + '.egofeat')) as fin:
                newfeat = ['0' for id in range(len(id2attr))]
                line = fin.readline().strip().split()
                for idx, value in enumerate(line):
                    newfeat[id2newid[idx]] = value

                fout.write(' '.join(newfeat) + '\n')

        with open(os.path.join(save_path, network_name + '.feat'), 'w') as fout:
            with open(os.path.join(path, network_name + '.feat')) as fin:
                newfeat = ['0' for id in range(len(id2attr))]
                for line in fin.readlines():
                    line = line.strip().split()
                    for idx, value in enumerate(line[1:]):
                        newfeat[id2newid[idx]] = value

                    fout.write(' '.join([line[0]] + newfeat) + '\n')


def prepare_facebook(path, combined_edges_path, save_path):
    '''
    prepare dataset for face_book, combine graphs together
    '''
    logger = get_logger()
    if not os.path.isdir(path):
        logger.error('%s is not a directory', path)
        return None

    # got id for each network
    network_names = set()
    for file_name in os.listdir(path):
        if file_name.split('.')[1] == 'egofeat':
            file_name = file_name.split('.')[0]
            network_names.add(file_name)

    id2attr, attr2id = combine_features(path, save_path, network_names)
    update_features(path, save_path, id2attr, attr2id, network_names)

    # copy combined path
    logger.info('copy combined path')
    shutil.copyfile(os.path.join(combined_edges_path, 'facebook_combined.txt'),
                    os.path.join(save_path, 'facebook_combined.txt'))

    return True


def build_graph_facebook(prepare_path, save_path=None, noise=None):
    '''
    build graph for facebook dataset
    '''
    logger = get_logger()
    graph = Graph(directed=False)

    # got id for each network
    network_names = set()
    for file_name in os.listdir(prepare_path):
        if file_name.split('.')[1] == 'egofeat':
            file_name = file_name.split('.')[0]
            network_names.add(file_name)

    for network_name in network_names:
        logger.info('building %s', network_name)
        with open(os.path.join(prepare_path, network_name + '.egofeat')) as fin:
            feature = [float(attr) for attr in fin.readline().strip().split()]
            node_id = network_name
            graph.add_node(node_id, feature)

        with open(os.path.join(prepare_path, network_name + '.feat')) as fin:
            for line in fin.readlines():
                line = line.strip().split()
                feature = [float(attr) for attr in line[1:]]
                graph.add_node(line[0], feature)

    logger.info('building edges')
    with open(os.path.join(prepare_path, 'facebook_combined.txt')) as fin:
        for line in fin.readlines():
            src, dst = line.strip().split()
            graph.add_edge(src, dst)

    graph.add_noise(noise)

    if save_path:
        logger.info('saving graph')
        with open(os.path.join(save_path, 'graph.pk'), 'wb') as f:
            pickle.dump(graph, f)

    return graph

def prepare_cora(path, save_path):
    '''
    prepare for label ids
    '''
    logger = get_logger()
    if not os.path.isdir(path):
        logger.error('%s is not a directory', path)
        return None

    logger.info('preparing for cora')
    label2id = {}
    id2label = []
    with open(os.path.join(path, 'cora.content')) as fin:
        for line in fin.readlines():
            line = line.strip().split()
            if line[-1] not in label2id:
                label2id[line[-1]] = len(id2label)
                id2label.append(line[-1])
    
    with open(os.path.join(save_path, 'label2id.pk'), 'wb') as fout:
        pickle.dump(label2id, fout)
    with open(os.path.join(save_path, 'id2label.pk'), 'wb') as fout:
        pickle.dump(id2label, fout)
    
    logger.info('done prepare for cora')

def build_graph_cora(path, save_path=None, noise=None):
    '''
    build graph for cora dataset
    '''
    logger = get_logger()
    graph = Graph(directed=False)

    logger.info('building nodes')
    with open(os.path.join(path, 'label2id.pk'), 'rb') as fin:
        label2id = pickle.load(fin)
    
    with open(os.path.join(path, 'cora.content')) as fin:
        for line in fin.readlines():
            line = line.strip().split()
            graph.add_node(line[0], [float(attr) for attr in line[1:-1]], 
                                     origin_attribute=None, label=label2id[line[-1]])

    logger.info('building edges')
    with open(os.path.join(path, 'cora.cites')) as fin:
        for line in fin.readlines():
            src, dst = line.strip().split()
            graph.add_edge(dst, src)

    graph.add_noise(noise)

    if save_path:
        logger.info('saving graph')
        with open(os.path.join(save_path, 'graph.pk'), 'wb') as f:
            pickle.dump(graph, f)

    return graph

def prepare_amazon(path, save_path):
    attrs = []
    logger = get_logger()
    with open(os.path.join(path, 'meta_Video_Games.json'), 'r') as fin:
        for idx, line in enumerate(fin, 1):
            if idx % 10000 == 0:
                logger.info('Reading categories %d', idx)
            line = eval(line)
            if 'related' not in line or 'also_bought' not in line['related']:
                continue
            for category in line['categories']:
                for s in category:
                    if s not in attrs:
                        attrs.append(s)
    
    with open(os.path.join(save_path, 'combined_feats.txt'), 'w')as fout:
        for idx, attr in enumerate(attrs):
            fout.write('%d %s\n'%(idx, attr))
    
    return True

def build_graph_amazon(path, save_path=None, noise=None):
    logger = get_logger()
    graph = Graph(directed=False)

    logger.info('building nodes')
    attrs = {}
    with open(os.path.join(path, 'combined_feats.txt'), 'r') as fin:
        for attr_num, line in enumerate(fin, 1):
            line = line.strip().split()
            attrs[' '.join(line[1:])] = int(line[0])
    
    valid = 0
    with open(os.path.join(path, 'meta_Video_Games.json'), 'r') as fin:
        for idx, line in enumerate(fin, 1):
            if idx % 10000 == 0:
                logger.info('building nodes %d', idx)
            line = eval(line)

            if 'related' not in line or 'also_bought' not in line['related']:
                continue
            valid += 1
            attr = [0 for i in range(attr_num)]
            for category in line['categories']:
                for s in category:
                    attr[attrs[s]] = 1
            
            graph.add_node(line['asin'], attribute=attr)
    logger.info('valid nodes: %d, total auctions: %d', valid, idx)

    edge_num = 0
    total_edges = 0
    with open(os.path.join(path, 'meta_Video_Games.json'), 'r') as fin:
        for idx, line in enumerate(fin, 1):
            if idx % 10000 == 0:
                logger.info('linking nodes %d', idx)
            line = eval(line)
            
            if 'related' in line and 'also_bought' in line['related']:
                for node in line['related']['also_bought']:
                    edge_num += graph.add_edge(line['asin'], node)
                    total_edges += 1
    logger.info('edge_num: %d total_edges: %d', edge_num, total_edges)
    
    graph.add_noise(noise)

    if save_path:
        logger.info('saving graph')
        with open(os.path.join(save_path, 'graph.pk'), 'wb') as f:
            pickle.dump(graph, f)

    return graph

def prepare_citeseer(path, save_path):
    '''
    prepare for label ids
    '''
    logger = get_logger()
    if not os.path.isdir(path):
        logger.error('%s is not a directory', path)
        return None

    logger.info('preparing for citeseer')
    label2id = {}
    id2label = []
    with open(os.path.join(path, 'citeseer.content')) as fin:
        for line in fin.readlines():
            line = line.strip().split()
            if line[-1] not in label2id:
                label2id[line[-1]] = len(id2label)
                id2label.append(line[-1])
    
    with open(os.path.join(save_path, 'label2id.pk'), 'wb') as fout:
        pickle.dump(label2id, fout)
    with open(os.path.join(save_path, 'id2label.pk'), 'wb') as fout:
        pickle.dump(id2label, fout)
    
    logger.info('done prepare for citeseer')

def build_graph_citeseer(path, save_path=None, noise=None):
    '''
    build graph for citeseer dataset
    '''
    logger = get_logger()
    graph = Graph(directed=False)

    logger.info('building nodes')
    with open(os.path.join(path, 'label2id.pk'), 'rb') as fin:
        label2id = pickle.load(fin)
    
    with open(os.path.join(path, 'citeseer.content')) as fin:
        for line in fin.readlines():
            line = line.strip().split()
            graph.add_node(line[0], [float(attr) for attr in line[1:-1]], 
                                     origin_attribute=None, label=label2id[line[-1]])

    logger.info('building edges')
    with open(os.path.join(path, 'citeseer.cites')) as fin:
        for line in fin.readlines():
            src, dst = line.strip().split()
            graph.add_edge(dst, src)

    graph.add_noise(noise)

    if save_path:
        logger.info('saving graph')
        with open(os.path.join(save_path, 'graph.pk'), 'wb') as f:
            pickle.dump(graph, f)

    return graph

def build_graph_pubmed(path, save_path=None, noise=None):
    '''
    build graph for pubmed dataset
    '''
    logger = get_logger()
    graph = Graph(directed=False)

    logger.info('building nodes')
    attr2id = {}
    with open(os.path.join(path, 'Pubmed-Diabetes.NODE.paper.tab'), 'r') as fin:
        line = fin.readline()
        line = fin.readline()
        line = line.strip().split()
        for idx, attr in enumerate(line, -1):
            if idx == -1: continue #label
            if attr.split(':')[1] == 'summary': continue
            attr2id[attr.split(':')[1]] = idx

        for line in fin:
            line = line.strip().split()
            attrs = [0.0 for i in range(len(attr2id))]
            for attr in line[2:-1]:
                attr = attr.split('=')
                attrs[attr2id[attr[0]]] = float(attr[1])
            label = int(line[1].split('=')[1])
            graph.add_node(line[0], attrs, origin_attribute=None, label=label)
    
    logger.info('building edges')
    with open(os.path.join(path, 'Pubmed-Diabetes.DIRECTED.cites.tab'), 'r') as fin:
        line = fin.readline()
        line = fin.readline()
        for line in fin:
            line = line.strip().split()
            src = line[1].split(':')[1]
            dst = line[3].split(':')[1]
            graph.add_edge(src, dst)

    graph.add_noise(noise)

    if save_path:
        logger.info('saving graph')
        with open(os.path.join(save_path, 'graph.pk'), 'wb') as f:
            pickle.dump(graph, f)

    return graph