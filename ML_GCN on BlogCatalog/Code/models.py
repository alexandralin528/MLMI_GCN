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
        
    def forward(self,Y_star, F_tilde,C_tilde):
        L = F.relu(self.gc1(Y_star,F_tilde))
        L = F.dropout(L, self.dropout, training = self.training)
        L = self.gc2(L,C_tilde)
        return F.log_softmax(L,dim=1)
        
        
 ## This GCN is a two layer GCN 
class Low_Layer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Low_Layer, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        
    def forward(self,X_star, E_tilde, A_tilde):
        N = F.relu(self.gc1(X_star, E_tilde))
        N = F.dropout(N, self.dropout, training = self.training)
        N = self.gc2(N, A_tilde)
        return F.sigmoid(N)
        
    