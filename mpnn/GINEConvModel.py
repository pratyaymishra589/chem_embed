import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.data import Data, Batch

class ReactionCenterMPNN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim=128, num_layers=4, out_dim=64):
        super(ReactionCenterMPNN, self).__init__()
        
        # 1. Initial Encoders: Project raw features to hidden_dim
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)
        
        # 2. Message Passing Layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # GINEConv requires an MLP to update the node embeddings after aggregating messages
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            
            # GINEConv automatically handles: MLP( (1+eps)*x_i + Sum( ReLU(x_j + edge_attr) ) )
            conv = GINEConv(nn=mlp, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # 3. Projection Head (Standard for Contrastive Learning like SimCLR/SupCon)
        # Maps the pooled embedding to the final contrastive space
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, data):
        # Unpack the PyG Data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        hop1_mask = data.hop1_mask

        # 1. Encode initial features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # 2. Message Passing (4 to 5 iterations)
        for conv, bn in zip(self.convs, self.batch_norms):
            # Calculate messages and update nodes
            x_out = conv(x, edge_index, edge_attr)
            x_out = bn(x_out)
            x_out = F.relu(x_out)
            
            # Residual connection (helps with training deeper GNNs)
            x = x + x_out 
            
        # 3. Masked Pooling (The crucial step for your task)
        # We slice the tensors to ONLY keep the atoms in the 1-hop reaction center
        x_masked = x[hop1_mask]
        batch_masked = batch[hop1_mask]
        
        # Aggregate the remaining atoms into a single vector per molecule in the batch
        # If batch_size is 32, `pooled` will be shape[32, hidden_dim]
        pooled = global_mean_pool(x_masked, batch_masked)
        
        # 4. Projection Head & Normalization
        # Contrastive learning works best when representations are L2 normalized 
        # so they sit on a unit hypersphere.
        out = self.projector(pooled)
        out = F.normalize(out, p=2, dim=-1)
        
        return out