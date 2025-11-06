from typing import Any, Optional
import numpy as np
from torch_geometric.data import Data as PyGData

import torch
import torch_geometric

from External.EGSST_PAPER.detector.efvit.efvit_backbone import EfficientViTLargeBackbone
from Models.base import BaseModel
from Models.EGSST.Components import EnchancedCNN


from Models.EGSST.transform import events_to_graph, TransformConfig


class EGSST(BaseModel):

    def __init__(
        self, *,
        target_size: tuple[int, int],
        detection_head_config: str,
        YOLOX: bool = False,
        Ecnn_flag: bool = False,
        ti_flag: bool = False,
    ):
        super().__init__()
        self.detection_head_config: str = detection_head_config
        self.YOLOX: bool = YOLOX
        self.Ecnn_flag: bool = Ecnn_flag
        self.ti_flag: bool = ti_flag
        # TODO: Load common components

        self.ECNN = None
        if self.Ecnn_flag:
            self.ECNN = EnchancedCNN(channels=32, target_size=target_size)

        self.MSLViT: EfficientViTLargeBackbone = EfficientViTLargeBackbone(
            width_list=[32, 64, 64, 128, 256],
            depth_list=[1, 1, 0, 1, 1],
            ti_flag=self.ti_flag,
        )

        self.detection_head = None
        if self.YOLOX:
            # TODO: FIND EXTERNAL IMPLEMENTATION
            raise NotImplementedError
        else:
            from External.EGSST_PAPER.detector import RTDETRHead
            self.detection_head = RTDETRHead(
                detection_head_config
            )

    def forward(self, x: torch_geometric.data.Data) -> torch.Tensor:
        if self.Ecnn_flag:
            x = self.ECNN(x)
        x = self.MSLViT(x)
        return self.detection_head(x)

    def data_transform(
    self,
    x: Any,
    cfg: Optional[TransformConfig] = None,
    label: Optional[int] = None
) -> PyGData:
    """
    Convert raw events -> torch_geometric.data.Data suitable for forward().
    Accepts:
      - PyG Data: returned as-is
      - numpy.ndarray or torch.Tensor with shape [N,4] (events [x,y,t,p])
      - dict with keys {'events', optional 'label'}
    """
    if isinstance(x, PyGData):
        return x


    if isinstance(x, dict):
        if 'events' not in x:
            raise ValueError("Dict input to data_transform must contain 'events' with an [N,4] array/tensor.")
        events = x['events']
        if label is None and 'label' in x:
            label = x['label']
    else:
        events = x

    # Validate shape
    if not (hasattr(events, 'shape') and len(events.shape) == 2 and events.shape[1] == 4):
        raise ValueError(f"Events must have shape [N,4], got {getattr(events, 'shape', None)}")

    # Build default config if none provided
    if cfg is None:
        # get model device robustly: 'cpu' or 'cuda'
        try:
            dev_str = str(next(self.parameters()).device)
            dev = 'cuda' if dev_str.startswith('cuda') else 'cpu'
        except StopIteration:
            dev = 'cpu'  # no parameters yet
        cfg = TransformConfig(
            beta=1000.0,          # scale seconds -> ms
            radius=10.0,          # tune per dataset
            min_nodes_subgraph=1, # raise to 8–16 after sanity checks
            max_num_neighbors=64,
            device=dev,
            ensure_undirected=True
        )


    graph = events_to_graph(events, cfg, label=label)
    return graph

