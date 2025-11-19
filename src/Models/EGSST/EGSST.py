import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.data import Data, Batch as PyGBatch
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_max

from External.EGSST_PAPER.detector.efvit.efvit_backbone import EfficientViTLargeBackbone
from Models.base import BaseModel
from Models.EGSST.Components import EnchancedCNN
from Models.utils import filter_connected_subgraphs, normalize_time, radius_graph_pytorch


class EGSST(BaseModel):

    def __init__(
        self, *,
        gcn_count: int,
        target_size: tuple[int, int],
        time_steps: int = 1,
        detection_head_config: str,
        YOLOX: bool = False,
        Ecnn_flag: bool = False,
        ti_flag: bool = False,
        task: str = "det",          # "det" = detection (current), "cls" = classification
        num_classes: int = 101,     # for N-Caltech
    ):

        super().__init__()
        self.gcn_count: int = gcn_count
        self.target_size: tuple[int, int] = target_size

        self.detection_head_config: str = detection_head_config
        self.YOLOX: bool = YOLOX
        self.Ecnn_flag: bool = Ecnn_flag
        self.ti_flag: bool = ti_flag

        self.task: str = task
        self.num_classes: int = num_classes


        self.GCNs = torch.nn.ModuleList()
        self.GCNs.append(GCNConv(4, 32))
        for _ in range(self.gcn_count):
            self.GCNs.append(GCNConv(32, 32))

        self.ECNN = None
        if self.Ecnn_flag:
            self.ECNN = EnchancedCNN(channels=32, target_size=target_size)

        self.MSLViT: EfficientViTLargeBackbone = EfficientViTLargeBackbone(
            width_list=[32, 64, 64, 128, 256],
            depth_list=[1, 1, 0, 1, 1],
            ti_flag=self.ti_flag,
            time_steps = time_steps
        )

        self.detection_head = None
        if self.task == "det":
            if self.YOLOX:
                # TODO: FIND EXTERNAL IMPLEMENTATION
                raise NotImplementedError
            else:
                from External.EGSST_PAPER.detector.rtdetr_header import RTDETRHead
                self.detection_head = RTDETRHead(
                    detection_head_config
                )
        else:
            # Classification head: global average pool + linear classifier
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.cls_proj = nn.LazyLinear(self.num_classes)  # will infer C on first forward


    def __wrap_into_dense(self, xy: torch.Tensor, features: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
        """
        Map node features to a dense [C, H, W] grid.

        target_size is assumed (W, H).
        xy contains integer pixel coordinates.
        """
        W, H = target_size  # width, height
        C = features.size(1)

        # Create dense tensor on same device/dtype as features
        dense = features.new_full((C, H * W), float('-inf'))

        # Clamp coordinates to valid range
        x = xy[:, 0].clamp(0, W - 1).long()
        y = xy[:, 1].clamp(0, H - 1).long()

        linear_idx = y * W + x

        for c in range(C):
            dense[c], _ = scatter_max(features[:, c], linear_idx, dim_size = H * W)

        dense = torch.where(torch.isinf(dense), torch.zeros_like(dense), dense)
        dense = dense.view(C, H, W)

        return dense

    
    def __take_feats(self, vit_out: torch.Tensor | list | tuple | dict) -> torch.Tensor:
        """
        Make backbone output into a single feature map tensor [B, C, H, W].
        """
        if isinstance(vit_out, torch.Tensor):
            return vit_out
        if isinstance(vit_out, (list, tuple)) and len(vit_out) > 0:
            return vit_out[-1]
        if isinstance(vit_out, dict) and len(vit_out) > 0:
            # take last added value
            return next(reversed(vit_out.values()))
        raise TypeError(f"Unexpected backbone output type: {type(vit_out)}")


    def forward(self, x: PyGBatch, **kwargs) -> torch.Tensor:
        graphs = []
        for graph in range(x.num_graphs):
            graph = x.get_example(graph)
            graphs.append((torch.cat([graph.pos, graph.x], dim=1), graph.edge_index))

        for gcn in self.GCNs:
            graphs_new = []
            for graph, edge_index in graphs:
                out = gcn(graph, edge_index)
                graphs_new.append((out, edge_index))
            graphs = graphs_new

        dense = []
        for idx, graph in enumerate(graphs):
            dense.append(self.__wrap_into_dense(x.get_example(idx).pos[:, :2].int(), graph[0], self.target_size)[None, :, :, :])
        dense = torch.cat(dense, dim = 0)  # [1, 32, H, W] for now

        if self.Ecnn_flag:
            dense = self.ECNN(dense)

        vit_out = self.MSLViT(dense)

        if self.task == "cls":
            feats = self.__take_feats(vit_out)           # [B, C, H, W]
            pooled = self.global_pool(feats)            # [B, C, 1, 1]
            logits = self.cls_proj(pooled.flatten(1))   # [B, num_classes]
            return logits
        else:
            return self.detection_head(vit_out, targets=kwargs.get("targets", None))

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