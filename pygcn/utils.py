import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import networkx as nx
from sklearn import preprocessing
import matplotlib.pyplot as plt


## load_data should be modified to load the blogCatalog data


def load_data(data_name): 
    print("Loading {} dataset...".format(data_name))
    edges_file = data_name + "/edges.csv"
    node_label_file = data_name + "/group-edges.csv"
    
    # We'll first dive into the group_edges.csv file in order to extract a list of the nodes along with their 
    # corresponding labels
    label_raw, nodes = [], []
    with open(node_label_file) as file_to_read: 
        while True:
            lines = file_to_read.readline()
            if not lines:
                break 
            node, label = lines.split(",")
            label_raw.append(int(label))
            nodes.append(int(node))
            
    # Now we have a list of nodes and a list of their labels
    # Since a node can have multiple labels, we should give each node a corresponding 39 lengthed vector that 
    # encodes 1 when the node has the label corresponding to the index and 0 otherwise.  
    label_raw = np.array(label_raw)
    nodes = np.array(nodes)
    labels = np.zeros((nodes.shape[0], 39))
    for l in range(1, 40, 1):
        indices = np.argwhere(label_raw == l).reshape(-1)
        n_l = nodes[indices]
        for n in n_l:
            labels[n-1][l-1] = 1
    
    # Now we can build our BlogCatalog graph using the file edges.csv 
    unique_nodes = np.unique(nodes)
    file_to_read = open(edges_file, 'rb')
    G = nx.read_edgelist(file_to_read, delimiter = ",", nodetype = int)
    
    # Let's now extract our adjacency matrix from the graph 
    A = nx.adjacency_matrix(G, nodelist = unique_nodes) # Already a symmetric matrix 
    A = sp.coo_matrix(A.todense())
    
    # Let's extract the feature matrix as well
    X = sp.csr_matrix(A)
    
    # As we saw in the paper, we need the normalized version of the adjacency matrix with the added self loops
    A = normalize(A + sp.eye(A.shape[0]))
    # X = normalize(X) --> Why do we need to do that ? 
    
    # Let's define the train, validation and test sets 
    indices = np.arange(A.shape[0]).astype('int32')
    # np.random.shuffle(indices)
    idx_train = indices[:A.shape[0] // 3]
    idx_val = indices[A.shape[0] // 3: (2 * A.shape[0]) // 3]
    idx_test = indices[(2 * A.shape[0]) // 3:]
    
    # Convert to tensors 
    X = torch.FloatTensor(np.array(X.todense()))
    labels = torch.LongTensor(labels)
    A = sparse_mx_to_torch_sparse_tensor(A)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    
    return A, X, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy_sample(output, labels):
    """ 
    output is of shape (N,C)
    Labels is of shape (N,C)
    Result : acc gives the accuracy computed according to the sample view
    """
    N = labels.shape[0]
    corr = np.sum(np.all(np.equal(output, labels), axis=1))
    # corr is the number of equal rows and thus the number of correctly classified samples
    acc = corr / N
    return acc 

def accuracy_sample_class(output, labels):
    """ 
    output is of shape (N,C)
    Labels is of shape (N,C)
    Result : acc gives the accuracy computed according to the sample-class view
    """
    N = labels.shape[0]
    C = labels.shape[1]
    corr = np.sum(np.equal(output, labels))
    # corr is the number of equal elements between labels and output and thus the number of correctly classified 
    # labels for each sample 
    acc = corr/(N*C)
    return acc



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)