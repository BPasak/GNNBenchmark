import torch
import torch_geometric
from torch_geometric.data import Data

from External.EGSST_PAPER.detector.efvit.efvit_backbone import EfficientViTLargeBackbone
from Models.base import BaseModel
from Models.EGSST.Components import EnchancedCNN
from Models.EGSST.transform import filter_connected_subgraphs, normalize_time, radius_graph_pytorch


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

        # self.detection_head = None
        # if self.YOLOX:
        #     # TODO: FIND EXTERNAL IMPLEMENTATION
        #     raise NotImplementedError
        # else:
        #     from External.EGSST_PAPER.detector.rtdetr_header import RTDETRHead
        #     self.detection_head = RTDETRHead(
        #         detection_head_config
        #     )

    def forward(self, x: Data) -> torch.Tensor:
        if self.Ecnn_flag:
            x = self.ECNN(x)
        x = self.MSLViT(x)
        return self.detection_head(x)

    def data_transform(
        self,
        x: Data,
        beta: float = 0.00001,
        radius: float = 5,
        min_nodes_subgraph: int = 100,
    ) -> Data:
        """
        Convert raw events in the format of torch_geometric.data.Data -> torch_geometric.data.Data suitable for forward().
        Accepts:
          - torch_geometric.data.Data
        """

        transformed: Data = x.clone()
        transformed.pos[:, 2] = normalize_time(transformed.pos[:, 2], beta=beta)
        edges = radius_graph_pytorch(transformed.pos, radius)
        transformed.edge_index = edges

        transformed = filter_connected_subgraphs(transformed, min_nodes = min_nodes_subgraph)

        return transformed

    def graph_update(
        self,
        x: torch_geometric.data.Data,
        event: tuple[float, float, float, float],
        **kwargs
    ) -> torch_geometric.data.Data:
        raise NotImplementedError