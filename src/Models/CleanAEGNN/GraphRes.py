import torch
from torch.nn import Linear
from torch.nn.functional import elu
from torch_geometric.data import Data as PyGData
from torch_geometric.nn import SplineConv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import radius_graph
from torch_geometric.transforms import Cartesian

from .MaxPooling import MaxPooling
from .MaxPoolingX import MaxPoolingX
from Models.utils import normalize_time, sub_sampling
from ..base import BaseModel
from torch_geometric.data import Batch as PyGBatch


class GraphRes(BaseModel):

    def __init__(
        self, input_shape: tuple[int, int, int],
        kernel_size: int, n: list[int], pooling_outputs: int,
        num_outputs: int, pooling_size=(16, 12),
        bias: bool = False, root_weight: bool = False,
    ):
        super(GraphRes, self).__init__()
        assert len(input_shape) == 3, "invalid input shape, should be (img_width, img_height, dim)"
        dim = int(input_shape[-1])

        self.conv1 = SplineConv(n[0], n[1], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm1 = BatchNorm(in_channels = n[1])
        self.conv2 = SplineConv(n[1], n[2], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm2 = BatchNorm(in_channels = n[2])

        self.conv3 = SplineConv(n[2], n[3], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm3 = BatchNorm(in_channels = n[3])
        self.conv4 = SplineConv(n[3], n[4], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm4 = BatchNorm(in_channels = n[4])

        self.conv5 = SplineConv(n[4], n[5], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm5 = BatchNorm(in_channels = n[5])
        self.pool5 = MaxPooling(pooling_size, transform = Cartesian(norm = True, cat = False))

        self.conv6 = SplineConv(n[5], n[6], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm6 = BatchNorm(in_channels = n[6])
        self.conv7 = SplineConv(n[6], n[7], dim = dim, kernel_size = kernel_size, bias = bias, root_weight = root_weight)
        self.norm7 = BatchNorm(in_channels = n[7])

        self.pool7 = MaxPoolingX([input_shape[0] // 4, input_shape[1] // 4], size = 16)
        self.fc = Linear(pooling_outputs * 16, out_features = num_outputs, bias = bias)

    def forward(self, data: PyGBatch, **kwargs) -> torch.Tensor:
        data.x = elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm1(data.x)
        data.x = elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm2(data.x)

        x_sc = data.x.clone()
        data.x = elu(self.conv3(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm3(data.x)
        data.x = elu(self.conv4(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm4(data.x)
        data.x = data.x + x_sc

        data.x = elu(self.conv5(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm5(data.x)
        data = self.pool5(data.x, pos = data.pos, batch = data.batch, edge_index = data.edge_index, return_data_obj = True)

        x_sc = data.x.clone()
        data.x = elu(self.conv6(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm6(data.x)
        data.x = elu(self.conv7(data.x, data.edge_index, data.edge_attr))
        data.x = self.norm7(data.x)
        data.x = data.x + x_sc

        x = self.pool7(data.x, pos = data.pos[:, :2], batch = data.batch)
        x = x.view(-1, self.fc.in_features)
        return self.fc(x)

    def data_transform(
        self, x: PyGData,
        n_samples: int = 10000, sampling: bool = True,
        beta: float = 0.5e-5, radius: float = 3.0,
        max_neighbors: int = 32, **kwargs
    ) -> PyGData:

        # Transform polarity from {-1,1} to {0,1}
        x.x = torch.where(x.x == -1., 0., x.x)

        window_us = 50 * 1000
        t = x.pos[x.num_nodes // 2, 2]
        index1 = torch.clamp(torch.searchsorted(x.pos[:, 2].contiguous(), t) - 1, 0, x.num_nodes - 1)
        index0 = torch.clamp(torch.searchsorted(x.pos[:, 2].contiguous(), t - window_us) - 1, 0, x.num_nodes - 1)
        num_nodes = x.num_nodes
        for key, item in x:
            if torch.is_tensor(item) and item.size(0) == num_nodes and item.size(0) != 1:
                x[key] = item[index0:index1, :]

        x = sub_sampling(x, n_samples = n_samples, sub_sample = sampling)

        # Re-weight temporal vs. spatial dimensions to account for different resolutions.
        x.pos[:, 2] = normalize_time(x.pos[:, 2], beta = beta)
        # Radius graph generation.
        x.edge_index = radius_graph(x.pos, r = radius, max_num_neighbors = max_neighbors)
        edge_attr = Cartesian(cat = False, max_value = 10.0)
        x.edge_attr = edge_attr(x).edge_attr
        return x

    def graph_update(
        self,
        x: PyGData,
        event: tuple[float, float, float, float],
        **kwargs
    ) -> PyGData:
        pass
