import torch

from torch_geometric.data import Data
from torch_geometric.nn.pool import max_pool, voxel_grid
from typing import Callable, List, Optional, Tuple, Union


class MaxPooling(torch.nn.Module):

    def __init__(self, size: List[int], transform: Callable[[Data, ], Data] = None):
        super(MaxPooling, self).__init__()
        self.voxel_size = list(size)
        self.transform = transform

    def forward(self, x: torch.Tensor, pos: torch.Tensor, batch: Optional[torch.Tensor] = None,
                edge_index: Optional[torch.Tensor] = None, return_data_obj: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.LongTensor, torch.Tensor, torch.Tensor], Data]:
        assert edge_index is not None, "edge_index must not be None"

        # Defensive handling: if there are no points (empty pos), avoid calling voxel_grid/grid ops
        # which assume non-empty inputs and will raise errors (see torch_cluster.grid).
        if pos is None or pos.numel() == 0 or pos.size(0) == 0:
            # Ensure all returned tensors exist and live on the same device as x.
            device = x.device if torch.is_tensor(x) else None
            # Create proper empty tensors with expected shapes:
            # - pos: (0, 3)
            # - batch: (0,)
            # - edge_attr: (0, 3) (pseudo-coordinates for SplineConv / Cartesian)
            pos_out = torch.empty((0, 3), device=device)
            batch_out = torch.empty((0,), dtype=torch.long, device=device)
            edge_attr_out = torch.empty((0, 3), device=device)

            data = Data(x=x, pos=pos_out, edge_index=edge_index, batch=batch_out)
            data.edge_attr = edge_attr_out

            if return_data_obj:
                return data
            else:
                return x, pos_out, batch_out, edge_index, edge_attr_out

        cluster = voxel_grid(pos[:, :2], batch=batch, size=self.voxel_size)
        data = Data(x=x, pos=pos, edge_index=edge_index, batch=batch)
        data = max_pool(cluster, data=data, transform=self.transform)  # transform for new edge attributes
        if return_data_obj:
            return data
        else:
            return data.x, data.pos, getattr(data, "batch"), data.edge_index, data.edge_attr

    def __repr__(self):
        return f"{self.__class__.__name__}(voxel_size={self.voxel_size})"
