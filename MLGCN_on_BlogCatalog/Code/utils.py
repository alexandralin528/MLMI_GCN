import numpy as np
import scipy.sparse as sp
import pandas as pd
import torch
import networkx as nx



# Let's define first a function to load the dataset 
# The role of this function is to load the dataset to memory and to extract the 
# necessary information we'll need for our model's input 

def load_data(data_name):
    print("Loading {} dataset...".format(data_name))
    edges_file = data_name + "/edges.csv"
    node_label_file = data_name + "/group-edges.csv"
    
    # We'll first dive into the group_edges.csv files to extract the nodes and their corresponding labels for the training
    label_raw, nodes = [], []
    with open(node_label_file) as file_to_read: 
        while True:
            lines = file_to_read.readline()
            if not lines:
                break 
            node, label = lines.split(",")
            label_raw.append(int(label))
            nodes.append(int(node))
    label_raw = np.array(label_raw)
    nodes = np.array(nodes)
    labels = np.zeros((nodes.shape[0], 39))
    for l in range(1, 40, 1):
        indices = np.argwhere(label_raw == l).reshape(-1)
        n_l = nodes[indices]
        for n in n_l:
            labels[n-1][l-1] = 1
            
    # In this section, we will build our node-node-label graph 
    unique_nodes = np.unique(nodes)
    label_nodes = label_raw + unique_nodes.shape[0]
    ## hereby are the nodes of the node-node-label graph 
    n_n_l_nodes = np.concatenate((unique_nodes, np.unique(label_nodes)))
    ## Let's create a csv file with all the edges to create the graph 
    ## Let's begin with the common nodes
    df = pd.DataFrame(list())
    df.to_csv(nnlg_file)
    f = open(nnlg_file, "w")
    file_to_read = open(edges_file, "r")
    f.writelines(file_to_read.readlines())
    
    ## Let's add the label edges
    a = np.dstack((label_nodes,nodes)).reshape(label_nodes.shape[0],2)
    e = [",".join(item)+"\n" for item in a.astype(str)]
    f.writelines(e)
    f.close()
    file_to_read.close()
    
    # Now let's create the node-node-label graph 
    nnl_graph = nx.read_edgelist(, delimiter = ",", nodetype = int)
    
    f.close()
    file_to_read.close()
    
    
    return label_nodes, nodes, e
    
    
    
    
    # In this section, we zill build our label-label-node graph
    
    
    
    
    
    
    
    
    # In this section, we will extract useful matrices and vectors from our graph to feed the model

    
    
    
    
    
    

    
    
    
    
    
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





def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

    
    