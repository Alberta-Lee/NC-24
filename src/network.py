import networkx as nx
import numpy as np
from sympy import *
from matplotlib  import pyplot as plt
import math

def init_node(*args, **kwargs):
    """
    initialize the nodes, and set the port for the nodes
    :param args: number of nodes
    :param kwargs:
    :return:
    """
    n = args[0]
    num = np.random.randint(12220, 50000)
    for i in range(1, n+1):
        G.add_node(i, host='127.0.0.1', port=num + i)

def add_neighbor_nodes(*args, **kwargs):
    """
    add neighbor nodes
    :param args: number of neighbor nodes
    :param kwargs:
    :return:
    """
    n = args[0]
    num_map = nx.to_numpy_matrix(G)
    
    for i in range(1, n+1):
        D = {}
        k = 1
        for attr in G[i].items():
            nid = attr[0]
            D[f'node{i}_neighbor{k}_host'] = G.nodes[nid]['host']
            D[f'node{i}_neighbor{k}_port'] = G.nodes[nid]['port']
            num_map[i-1, nid-1] = k
            k += 1
        G.add_node(i, **D)
        G.add_node(i, num_of_neighbors=len(G[i].items()))
    
    return num_map

def local_degree(eps_deg = 1):
    p = nx.to_numpy_matrix(G)
    n = len(G.nodes())
    w_fdla = np.zeros((n, n))
    deg = p.sum(axis = 1)
    
    for i in range(n):
        for j in range(n):
            if p[i, j] == 1:
                w_fdla[i][j] = 1 / (max(deg[i], deg[j]) + eps_deg)
    y = (np.ones((n, 1)) - (np.mat(w_fdla)).sum(axis=1)).A.reshape(w_fdla.shape[1])
    w_net = np.diag(y) + w_fdla
    eig_w = np.sort(np.linalg.eigvals(np.mat(w_net)))
    alpha = math.sqrt(1 - eig_w[1])
    eta = 1 - abs(eig_w[1])
    return alpha, eta

def local_beta():
    p = nx.to_numpy_matrix(G)
    n = len(G.nodes())
    I = np.identity(n)
    p = I - p
    beta = np.linalg.norm(p)
    return beta

def local_gamma():
    alpha, rho = local_degree()
    beta = local_beta()
    delta = 0.8
    gamma = rho**2 / (16*rho + rho**2 + 4*beta**2 + 2*rho*beta**2 - 8*rho*delta)
    return gamma

def get_0_1_array(array, rate=0.2):
    zeros_num = int(array.size * rate)
    new_array = np.ones(array.size)
    new_array[:zeros_num] = 0
    np.random.shuffle(new_array)
    re_array = new_array.reshape(array.shape)
    return re_array 

def matrix_to_graph(options, rate=0.5):
    global G
    G = nx.Graph()

    # create adjacency matrix
    if options == 'ring':
        graph = nx.cycle_graph(5)   # 5 nodes
        graph_mat = nx.adjacency_matrix(graph)
        matrix = graph_mat.todense().tolist()
    
    elif options == 'random':
        arr = np.ones((5, 5))
        matrix = get_0_1_array(arr, rate=0.9)
        matrix[range(5), range(5)] = 0
    
    elif options == 'all_connection':
        matrix = np.ones((5, 5))
        matrix[range(5), range(5)] = 0

    n = len(matrix)
    init_node(n)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if matrix[i-1][j-1] == 1:
                G.add_edge(i, j)
    print(nx.to_numpy_matrix(G))
    local_gamma()
    num_map = add_neighbor_nodes(n)
    return num_map, G


if __name__ == '__main__':
    num_map, G = matrix_to_graph('ring')
    xx = num_map[1][:]
    for i in xx:
        print(i)
    yy = xx.tolist()

    for n, nbrs in G.adjacency():
        print(n, nbrs)

    for i in range(1, len(G.nodes())+1):
        print(G.nodes[i])
    print(local_degree())
    nx.draw_spring(G, cmap=plt.get_cmap('jet'), node_color='b', alpha=0.6, node_size=30, with_labels=False)
    plt.show()