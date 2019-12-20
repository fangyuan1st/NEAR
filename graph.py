from collections import Iterable
import random
import copy
from util import get_logger


class Graph():
    '''
    a Graph contains 
        graph: a list of adjacency-table by id
        ingraph: a list of adjacency-table by id
        origin_attribute: a list from node to attribute vector(a list) by id
        attribute: a list from node to attribute vector(a list) by id noised
        label: a list from node_id to label_id
        node2id: a dict from node label to id
        id2node: a list from id to node label
    '''

    def __init__(self, directed=True):
        self.__directed = directed
        self.graph = []
        self.ingraph = []
        self.attribute = []
        self.origin_attribute = []
        self.label = {}
        self.node2id = {}
        self.id2node = []
        self.edges = []
    
    def is_directed(self):
        return self.__directed

    def node_num(self):
        '''
        number of nodes in the graph
        '''
        return len(self.id2node)

    def attr_num(self):
        '''
        number of attribute for each node
        '''
        if len(self.attribute):
            return len(self.attribute[0])
        return None

    def add_node(self, node_label, attribute=None, origin_attribute=None, label=None):
        '''
        Args:
            node_label should be an integer
        '''
        logger = get_logger()
        if node_label not in self.node2id:
            self.node2id[node_label] = len(self.id2node)
            self.id2node.append(node_label)
            self.graph.append([])
            self.ingraph.append([])
            if attribute:
                if len(self.attribute) and len(self.attribute[0]) != len(attribute):
                    logger.error('attribute length different')
                self.attribute.append(attribute)
                if origin_attribute:
                    self.origin_attribute.append(copy.deepcopy(origin_attribute))
                else:
                    self.origin_attribute.append(copy.deepcopy(attribute))
            
            self.label[self.node2id[node_label]] = label# can be None

        else:
            logger.warning('node %s added before', node_label)
            if self.attribute[self.node2id[node_label]] != attribute:
                logger.warning(
                    'node %s has different attribute will be combined', node_label)
                feature = self.attribute[self.node2id[node_label]]
                origin_feature = self.origin_attribute[self.node2id[node_label]]
                for idx, value in enumerate(feature):
                    if value != attribute[idx]:
                        if value == 0:
                            feature[idx] = attribute[idx]
                            origin_feature[idx] = attribute[idx]
                        elif attribute[idx] != 0:
                            logger.error('node %s:%d attribute conflict: %lf: %lf',
                                         node_label, idx, value, attribute[idx])
                            return
            
            if self.label[self.node2id[node_label]] != label:
                logger.error('node %s:label conflict: %s: %s', node_label, label, self.label[self.node2id[node_label]])
                return

    def add_edge(self, src, dst):
        '''
        src and dst are both node labels
        return
        1 if success
        0 if not
        '''
        logger = get_logger()
        if src not in self.node2id:
            #logger.error('node %s has not been added', src)
            return 0
        if dst not in self.node2id:
            #logger.error('node %s has not been added', dst)
            return 0

        src_id = self.node2id[src]
        dst_id = self.node2id[dst]
        if src_id == dst_id:
            logger.error('src and dst are same')
            return 0
        if self.__directed:
            if dst_id not in self.graph[src_id]:
                self.graph[src_id].append(dst_id)
                self.ingraph[dst_id].append(src_id)
                self.edges.append((src_id, dst_id))
        else:
            if dst_id not in self.graph[src_id]:
                self.graph[src_id].append(dst_id)
                self.ingraph[dst_id].append(src_id)
                self.graph[dst_id].append(src_id)
                self.ingraph[src_id].append(dst_id)
                self.edges.append((src_id, dst_id))
        
        return 1

    def add_noise(self, noise):
        '''
        add noise to attribute
        Args:
            noise: probability to change from 0->1
        '''
        random.seed(0)
        logger = get_logger()
        if noise:
            logger.info('adding noise')
            for feature in self.attribute:
                for idx, value in enumerate(feature):
                    if value == 0:
                        if random.random() < noise:
                            feature[idx] = 1
            logger.info('finish adding noise')
    
    def indegree(self, nodes=None):
        '''
        nodes should be nodes ids
        '''
        if isinstance(nodes, Iterable):
            return [len(self.ingraph[v]) for v in nodes]
        else:
            return len(self.ingraph[nodes])
    
    def outdegree(self, nodes=None):
        '''
        nodes should be nodes ids
        '''
        if isinstance(nodes, Iterable):
            return [len(self.graph[v]) for v in nodes]
        else:
            return len(self.graph[nodes])

    def degree(self, nodes=None):
        '''
        nodes should be nodes ids
        '''
        if isinstance(nodes, Iterable):
            if self.__directed:
                return [len(self.graph[v]) + len(self.ingraph[v]) for v in nodes]
            else:
                return [len(self.graph[v]) for v in nodes]
        else:
            if self.__directed:
                return len(self.graph[nodes]) + len(self.ingraph[nodes])
            else:
                return len(self.graph[nodes])

    def random_walk(self, path_length, start=None, alpha=0, rand=random.Random()):
        '''
        path formed with node ids
        '''
        random.seed(0)
        if start:
            path = [start]
        else:
            path = [rand.choice(len(self.id2node))]

        while len(path) < path_length:
            cur = path[-1]
            if len(self.graph[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(self.graph[cur]))
                else:
                    path.append(path[0])
            else:
                break

        return path

    def sort_by_degree(self, node_ids, reverse=False):
        '''
        return a list of node_ids sorted by degree
        '''
        logger = get_logger()
        logger.info('sorting by degree')
        degrees = self.degree(node_ids)
        res = sorted(enumerate(degrees), key=lambda x: x[1], reverse=reverse)
        return [x[0] for x in res]
    
    def reprojection(self, projection):
        '''
        Args:
            projection is a list or dict, with new_node_id = projection[old_node_id]
        Return:
            True if successful, False if not
        '''
        logger = get_logger()
        logger.info('reprojecting node to node_id')
        if len(set(projection)) != len(projection):
            logger.info('projection not valid')
            return False
        
        num_nodes = self.node_num()
        newgraph = [None for i in range(num_nodes)]
        newingraph = [None for i in range(num_nodes)]
        newattr = [None for i in range(num_nodes)]
        new_originattr = [None for i in range(num_nodes)]
        new_id2node = [None for i in range(num_nodes)]
        new_label = {}
        for old_node_id, new_node_id in enumerate(projection):
            newgraph[new_node_id] = self.graph[old_node_id]
            newingraph[new_node_id] = self.ingraph[old_node_id]
            for idx, node_id in enumerate(newgraph[new_node_id]):
                newgraph[new_node_id][idx] = projection[node_id]
            for idx, node_id in enumerate(newingraph[new_node_id]):
                newingraph[new_node_id][idx] = projection[node_id]
            newattr[new_node_id] = self.attribute[old_node_id]
            new_originattr[new_node_id] = self.origin_attribute[old_node_id]
            new_id2node[new_node_id] = self.id2node[old_node_id]
            new_label[new_node_id] = self.label[old_node_id]
            self.node2id[self.id2node[old_node_id]] = new_node_id

        for idx, edge in enumerate(self.edges):
            new_edge0 = projection[edge[0]]
            new_edge1 = projection[edge[1]]
            self.edges[idx] = (new_edge0, new_edge1)
        
        self.graph = newgraph
        self.ingraph = newingraph
        self.attribute = newattr
        self.origin_attribute = new_originattr
        self.id2node = new_id2node
        self.label = new_label

        logger.info('reprojection finish')
        return True
