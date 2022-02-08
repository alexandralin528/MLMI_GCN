import torch.nn as nn
import torch.nn.functional as F
from Code.layers import GraphConvolution
import torch

# In this file, we'll define two GCNs : the high layer GCN and the low layer GCN
## This GCN is a two layer GCN 
class High_Layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(High_Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fc1 = nn.Linear(nclass, nfeat)
    
   
    def forward(self,Y_star, F_tilde,C_tilde):
        L = F.relu(self.gc1(Y_star,F_tilde))
        L = F.dropout(L, self.dropout, training = self.training)
        L = self.gc2(L,C_tilde)
        Y_new = L
        L = self.fc1(L)
        return Y_new
        
        
 ## This GCN is a two layer GCN 
class Low_Layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Low_Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.fc1 = nn.Linear(nclass, nfeat)
        
        
    def forward(self,X_star, E_tilde, A_tilde):
        N = F.relu(self.gc1(X_star, E_tilde))
        N = F.dropout(N, self.dropout, training = self.training)
        N = self.gc2(N, A_tilde)
        X_new = N
        N = self.fc1(N)
        return X_new
        
class Global_Layer(torch.nn.Module):
    def __init__(self):
        super(Global_Layer, self).__init__()
        self.weight = torch.nn.Parameter(torch.FloatTensor(39, 10312))

    def forward(self, input):
        support = torch.mm(input, self.weight)

        return support
    