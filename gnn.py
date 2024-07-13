import numpy as np
import random
import torch

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
    def __init__(self, nn: torch.nn.Module):
        # Message passing with "max" aggregation.
       super().__init__(aggr='max')
       self.nn = nn

    def forward(
            self, x: Tuple[torch.Tensor, ...], pos: Tuple[torch.Tensor, ...], edge_index: torch.Tensor
        ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_j: torch.Tensor, pos_j: torch.Tensor, pos_i: torch.Tensor) -> torch.Tensor:
        # Translate all positions to the coordinate frame of the centroids
        input = pos_j - pos_i

        if x_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([x_j, input], dim=-1)

        return self.nn(input)  # Apply our final neural network.


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetLayer(nn)

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
        x = self.conv((x, x_dest), (pos, pos[idx]), edge_index)

        # Get the centroid positions for each batch
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Concatenate all the node features and positions and run though a neural network
        x = self.nn(torch.cat([x, pos], dim=1))
        # Pool across all the node features in the batch to produce graph embeddings of dim [num_graphs, F_x]
        x = global_max_pool(x, batch)
        # Create an empty tensor of positions for each graph embedding we got at the previous step
        pos = pos.new_zeros((x.size(0), 3))
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
        self.sa1_module = SAModule(ratio=0.2, r=0.2, nn=MLP([3, 32, 32]))
        # Sample 25 % of the downsampled points, group within the radius of 0.4 (since the points are more sparse now)
        # and encode them into a 32-dim vector
        self.sa2_module = SAModule(ratio=0.25, r=0.4, nn=MLP([32 + 3, 32, 64]))
        # Take each point positions and features, encode them into a 64-dim vector and then max-pool across all graphs
        self.sa3_module = GlobalSAModule(MLP([64 + 3, 64, 128]))

        # Insert LSTM -> dim = 128
        self.lstm = torch.nn.LSTM(128, 128, 2 * batch_size, batch_first=True)

        # Perform upsampling and feature propagation
        # Interpolate output features from sa3_module and concatenate with the sa2_module output features
        # Input features are 64-dim from sa3_module and 32-dim from sa2_module
        self.fp3_module = FPModule(2, MLP([128 + 64, 64]))
        # Interpolate upsampled features from fp3_module and concatenate with sa1_module output features
        # Input features are 32-dim from fp3_module and 16-dim from sa1_module
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