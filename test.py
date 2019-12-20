import pickle
import numpy as np
from graph import Graph

with open('/home/xiaotong/data/ego-facebook/graph.pk', 'rb') as fin:
    graph = pickle.load(fin)

total_attr = 0
total_hop1 = 0
total_hop1_sim = 0
for node in range(graph.node_num()):
    attr = np.array(graph.attribute[node])
    hop1_attr = np.array([graph.attribute[hop1] for hop1 in graph.graph[node]])

    total_attr += np.sum(attr)
    total_hop1 += len(hop1_attr)
    total_hop1_sim += np.dot(attr,np.sum(hop1_attr, axis=0))

result = {'avg_attr': total_attr/graph.node_num(),
          'total_hop1': total_hop1,
          'avg_hop1_sim': total_hop1_sim/total_hop1,
          'attr_num': graph.attr_num()}

print(result)

total_hop2 = 0
total_hop2_sim = 0
hop2_relation = []
for node in range(graph.node_num()):
    temp = []
    for hop1 in graph.graph[node]:
      temp.extend([hop2 for hop2 in graph.graph[hop1]])
    temp = list(set(temp))
    total_hop2 += len(temp)
    hop2_relation.append(temp)

print('total_hop2: %d'%total_hop2)

for node in range(graph.node_num()):
    if node % 100 == 0:
        print(node, len(hop2_relation[node]))
    attr = np.array(graph.attribute[node])
    hop2_attr = np.zeros(graph.attr_num())
    for hop2 in hop2_relation[node]:
        hop2_attr += graph.attribute[hop2]
    #hop2_attr = np.array([graph.attribute[hop2] for hop2 in hop2_relation[node]])
    #total_hop2_sim += np.dot(attr, np.sum(hop2_attr, axis=0))
    total_hop2_sim += np.dot(attr, hop2_attr)

print('avg_hop2_sim: %lf'%(total_hop2_sim/total_hop2))


