import numpy as np
import random
import torch

from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from torch_geometric.nn.pool import fps, radius, global_max_pool
from torch_geometric.nn.unpool import knn_interpolate
from torch_geometric.nn import MessagePassing, MLP
from typing import Tuple, Optional

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PointNetLayer(MessagePassing):
    def __init__(self, nn_msg_x: torch.nn.Module, nn_msg_pos: torch.nn.Module, nn_upd_x: torch.nn.Module):
        # Message passing with "max" aggregation.
        super().__init__(aggr='add')
        self.nn_msg_x = nn_msg_x
        self.nn_msg_pos = nn_msg_pos
        self.nn_upd_x = nn_upd_x

    def forward(
            self, x: Tuple[torch.Tensor, ...], pos: Tuple[torch.Tensor, ...], edge_index: torch.Tensor
        ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
        # Translate all positions to the coordinate frame of the centroids
        dist = torch.norm(pos_j - pos_i, dim=-1, keepdim=True)

        if x_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            x = torch.cat([x_j, dist], dim=-1)
        else:
            x = dist

        # Get the edge embeddings
        edge_emb = self.nn_msg_x(x)
        # Calculate scalar weights for each relative position
        pos_msg_weights = self.nn_msg_pos(edge_emb)
        # Calculate the positional messages
        pos_msg = (pos_j - pos_i) * pos_msg_weights
        return edge_emb, pos_msg

    def aggregate(self, inputs, index):
        # Get all the messages for the node features
        aggr_out_x = inputs[0]
        # Get all the messages for the node coordinates
        aggr_out_pos = inputs[1]
        # Aggregate all node features
        aggr_out_x = scatter(src=aggr_out_x, index=index, dim=self.node_dim, reduce=self.aggr)
        # Aggregate all node coordinates (Note! We use mean hear, as in the paper)
        aggr_out_pos = scatter(src=aggr_out_pos, index=index, dim=self.node_dim,reduce="mean")
        return aggr_out_x, aggr_out_pos
    
    def update(self, aggr_out, x, pos):
        # Get aggregated node features and coordinates
        aggr_out_x, aggr_out_pos = aggr_out[0], aggr_out[1]
        # Concatenate node features with the aggregated ones
        if x[1] is not None:
            upd_out_x = torch.cat([x[1], aggr_out_x], dim=-1)
        else:
            upd_out_x = aggr_out_x
        # Update positions simply adding the aggregated positions (this ensures equivariance)
        upd_out_pos = pos[1] + aggr_out_pos
        return self.nn_upd_x(upd_out_x), upd_out_pos

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn_msg_x, nn_msg_pos, nn_upd_x):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetLayer(nn_msg_x, nn_msg_pos, nn_upd_x)

    def forward(self, x: Optional[torch.Tensor], pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Apply FPS to get the centroids for each graph in the batch
        idx = fps(pos, batch, ratio=self.ratio)

        # Group the neighbours of the centroids for each graph in the batch
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)

        # See if we have any hidden node features that we received from the previous layer
        x_dest = None if x is None else x[idx]

        # Apply PointNetLayer using the features of the neighbouring nodes and the centroids
        # The output is the updated node features for the centroids
        # We feed tuples to differentiate between the features of the neighbouring nodes and the centroids
        x, pos = self.conv((x, x_dest), (pos, pos[idx]), edge_index)

        # Get the centroid positions for each batch
        batch = batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn_x):
        super().__init__()
        self.nn_x = nn_x
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Concatenate all the node features and positions and run though a neural network
        x = self.nn_x(x)
        # Pool across all the node features in the batch to produce graph embeddings of dim [num_graphs, F_x]
        x = global_max_pool(x, batch)
        pos = global_max_pool(pos, batch)
        # Create a new batch tensor for each graph embedding
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch
   

class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(
            self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor, x_skip: torch.Tensor,
            pos_skip: torch.Tensor, batch_skip: torch.Tensor
            ) -> Tuple[torch.Tensor, ...]:
            # Perform the interpolation
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)

        # Check if there was any previous SA layer output to concatenate with
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)

        # Encode the features with a neural network
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNetPP(torch.nn.Module):
    def __init__(self, num_coords, batch_size):
        super().__init__()
        seed(12345)
        self.num_coords = num_coords 
        self.batch_size = batch_size
        # Input channels account for both `pos` and node features.
        # Perform the downsampling and feature aggregation
        # Sample 20% of input points, group them within the radius of 0.2 and encode the features into a 16-dim vector
        nn_msg_x = MLP([1, 32, 32])
        nn_msg_pos = MLP([32, 32, 1])   
        nn_upd_x = MLP([32, 32, 32])
        self.sa1_module = SAModule(ratio=0.2, r=0.2, nn_msg_x=nn_msg_x, nn_msg_pos=nn_msg_pos, nn_upd_x=nn_upd_x)
        # Sample 25 % of the downsampled points, group within the radius of 0.4 (since the points are more sparse now)
        # and encode them into a 32-dim vector
        nn_msg_x = MLP([32 + 1, 32, 64])
        nn_msg_pos = MLP([64, 64, 1])   
        nn_upd_x = MLP([64 + 32, 64, 64])
        self.sa2_module = SAModule(ratio=0.25, r=0.4, nn_msg_x=nn_msg_x, nn_msg_pos=nn_msg_pos, nn_upd_x=nn_upd_x)
        # Take each point positions and features, encode them into a 64-dim vector and then max-pool across all graphs
        nn_x = MLP([64, 64, 128])
        self.sa3_module = GlobalSAModule(nn_x=nn_x)

        # Insert LSTM -> dim = 128
        self.lstm = torch.nn.LSTM(128, 128, 2 * batch_size, batch_first=True)

        # Perform upsampling and feature propagation
        # Interpolate output features from sa3_module and concatenate with the sa2_module output features
        self.fp3_module = FPModule(2, MLP([128 + 64, 64]))
        # Interpolate upsampled features from fp3_module and concatenate with sa1_module output features
        self.fp2_module = FPModule(3, MLP([64 + 32, 32]))
        # Interpolate upsampled output features from fp2_module and encode them into a 128-dim vector
        self.fp1_module = FPModule(3, MLP([32, 128]))


        # Apply the final MLP network to perform label segmentation for each point, using their propagated features
        self.mlp = MLP([128, 64, num_coords], dropout=0.5, norm=None)


    def forward(self, input_points, input_tindex):
        data_list = []
        for i in range(input_points.shape[0]):
            input_points_0 = input_points[i, input_tindex[i] == 0]
            input_points_1 = input_points[i, input_tindex[i] == 1]

            pad_len = self.batch_size - input_points_0.shape[0] % self.batch_size
            input_points_0 = torch.cat([input_points_0, torch.zeros(pad_len, 3)], dim=0)

            pad_len = self.batch_size - input_points_1.shape[0] % self.batch_size
            input_points_1 = torch.cat([input_points_1, torch.zeros(pad_len, 3)], dim=0)
            
            data_list.append(Data(x=None, pos=input_points_0))
            data_list.append(Data(x=None, pos=input_points_1))
        data = Batch.from_data_list(data_list)
        del data_list
        del input_points_0, input_points_1, input_points

        sa0_out = (data.x, data.pos, data.batch)
        sa1_out = self.sa1_module(*sa0_out) # x, pos, batch after the 1st local SA layer
        sa2_out = self.sa2_module(*sa1_out) # x, pos, batch after the 2nd local SA layer
        sa3_out = self.sa3_module(*sa2_out) # x, pos, batch after the 3rd global SA layer (pos are all zeros here!)

        # LSTM
        lstm_out = self.lstm(sa3_out[0].unsqueeze(0))[0].squeeze(0) # Would be nice to doublecheck

        # Replace sa3_out learned features with the LSTM output
        sa3_out = (lstm_out, sa3_out[1], sa3_out[2])

        fp3_out = self.fp3_module(*sa3_out, *sa2_out) # x, pos, batch after upsampling in the 3rd FP layer
        fp2_out = self.fp2_module(*fp3_out, *sa1_out) # x, pos, batch after upsampling in the 2nd FP layer
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out) # x - node embeddings for each point in the original point clouds

        # Generate final label predictions for each data point in each batch
        return self.mlp(x).view(self.batch_size, -1, self.num_coords) # Dim = batch x Num_nodes x num_coords