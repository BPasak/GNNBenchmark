import torch
import torch_geometric

from External.EGSST_PAPER.detector.efvit.efvit_backbone import EfficientViTLargeBackbone
from Models.base import BaseModel
from Models.EGSST.Components import EnchancedCNN


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

    def data_transform(self, x: torch_geometric.data.Data) -> torch_geometric.data.Data:
        pass # TODO: Implement Data Transform
