import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool_x, voxel_grid
from typing import List, Optional, Tuple, Union


class MaxPoolingX(torch.nn.Module):

    def __init__(self, voxel_size: List[int], size: int):
        super(MaxPoolingX, self).__init__()
        self.voxel_size = voxel_size
        self.size = size

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        # Defensive: some samples may have empty pos tensors (no events after sampling/filtering).
        # Calling voxel_grid on an empty tensor triggers a C++ reshape error in torch_cluster.
        # Return a device-consistent empty tensor instead of calling voxel_grid in that case.
        if pos is None or pos.numel() == 0:
            # preserve dtype/device and channel dimension
            if x is None:
                # nothing to do, return an empty tensor
                return torch.empty((0,), device=pos.device if pos is not None else None)
            # x has shape [N, C]; return empty [0, C] on same device/dtype
            return x.new_empty((0, x.size(1)))

        cluster = voxel_grid(pos, batch=batch, size=self.voxel_size)
        x, _ = max_pool_x(cluster, x, batch, size=self.size)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size}, size={self.size})"
