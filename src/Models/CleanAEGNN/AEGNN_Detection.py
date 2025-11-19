##borrowed from AEGNN repo
"""Partly copied from rpg-asynet paper: https://github.com/uzh-rpg/rpg_asynet"""
from typing import Tuple

import torch
from torch_geometric.data import Batch as PyGBatch

from utils.bounding_box import crop_to_frame, non_max_suppression
from utils.yolo import yolo_grid
from .GraphRes import GraphRes


class AEGNN_Detection(GraphRes):
    def __init__(
        self, input_shape: tuple[int, int, int],
        kernel_size: int, n: list[int], pooling_outputs: int,
        num_classes: int, num_bounding_boxes: int = 1,
        pooling_size=(16, 12), cell_map_shape = (8, 6),
        bias: bool = False, root_weight: bool = False
    ):

        self.num_classes = num_classes
        self.num_bounding_boxes = num_bounding_boxes
        self.cell_map_shape = cell_map_shape

        self.num_outputs_per_cell = num_classes + num_bounding_boxes * 5  # (x, y, width, height, confidence)
        num_outputs = self.num_outputs_per_cell * self.cell_map_shape[0] * self.cell_map_shape[1]

        self.cell_x_shift = (torch.tensor(list(range(self.cell_map_shape[0])), requires_grad = False)/self.cell_map_shape[0])[None, :, None, None]
        self.cell_y_shift = (torch.tensor(list(range(self.cell_map_shape[1])), requires_grad = False)/self.cell_map_shape[1])[None, None, :, None]

        super(AEGNN_Detection, self).__init__(
            input_shape = input_shape,
            kernel_size = kernel_size, n = n, pooling_outputs = pooling_outputs,
            num_outputs = num_outputs, pooling_size = pooling_size,
            bias = bias, root_weight = root_weight,
        )
        
    def forward(self, data: PyGBatch, **kwargs) -> torch.Tensor:
        out = super(AEGNN_Detection, self).forward(data, **kwargs)
        out = out.view(-1, *self.cell_map_shape, self.num_outputs_per_cell)
        parsed_out = self.parse_output(out)
        center_x = parsed_out[0]/self.cell_map_shape[0] + self.cell_x_shift.to(out.device)
        center_y = parsed_out[1]/self.cell_map_shape[1] + self.cell_y_shift.to(out.device)

        out_dict = {
            "pred_logits": parsed_out[5].reshape(center_x.shape[0], -1, self.num_classes),
            "pred_boxes": torch.cat(
                [center_x, center_y, parsed_out[3], parsed_out[2]],
                dim = -1
            ).reshape(center_x.shape[0], -1, 4),
        }

        return out_dict

    ###############################################################################################
    # Parsing #####################################################################################
    ###############################################################################################
    def parse_output(
        self, model_output: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nr_bbox = self.num_bounding_boxes

        x_norm_rel = torch.clamp(model_output[..., 0:nr_bbox], min = 0)  # Center x
        y_norm_rel = torch.clamp(model_output[..., nr_bbox:nr_bbox * 2], min = 0)  # Center y
        w_norm_sqrt = torch.clamp(model_output[..., nr_bbox * 2:nr_bbox * 3], min = 0)  # Height
        h_norm_sqrt = torch.clamp(model_output[..., nr_bbox * 3:nr_bbox * 4], min = 0)  # Width
        y_confidence = torch.sigmoid(model_output[..., nr_bbox * 4:nr_bbox * 5])  # Object Confidence
        y_class_scores = model_output[..., nr_bbox * 5:]  # Class Score

        return x_norm_rel, y_norm_rel, w_norm_sqrt, h_norm_sqrt, y_confidence, y_class_scores

    @staticmethod
    def parse_gt(
        gt_bbox: torch.Tensor, input_shape: torch.Tensor, cell_map_shape: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cell_corners = yolo_grid(input_shape, cell_map_shape)
        cell_shape = input_shape / cell_map_shape

        gt_cell_corner_offset_x = gt_bbox[..., 0, None] - cell_corners[None, :, 0, 0]
        gt_cell_corner_offset_x[gt_cell_corner_offset_x < 0] = 9999999
        gt_cell_corner_offset_x, gt_cell_x = torch.min(gt_cell_corner_offset_x, dim = -1)

        gt_cell_corner_offset_y = gt_bbox[..., 1, None] - cell_corners[None, 0, :, 1]
        gt_cell_corner_offset_y[gt_cell_corner_offset_y < 0] = 9999999
        gt_cell_corner_offset_y, gt_cell_y = torch.min(gt_cell_corner_offset_y, dim = -1)

        gt_cell_corner_offset = torch.stack([gt_cell_corner_offset_x, gt_cell_corner_offset_y], dim = -1)
        gt_cell_corner_offset_norm = gt_cell_corner_offset / cell_shape[None, :].float()

        gt_bbox_shape = torch.stack([gt_bbox[..., 2], gt_bbox[..., 3]], dim = -1)
        gt_bbox_shape_norm_sqrt = torch.sqrt(gt_bbox_shape / input_shape.float())
        return gt_cell_corner_offset_norm, gt_bbox_shape_norm_sqrt, gt_cell_x, gt_cell_y

    ###############################################################################################
    # YOLO Detection ##############################################################################
    ###############################################################################################
    def detect(self, model_output: torch.Tensor, threshold: float = None) -> torch.Tensor:
        """Computes the detections used in YOLO: https://arxiv.org/pdf/1506.02640.pdf"""
        cell_map_shape = torch.tensor(model_output.shape[1:3], device = model_output.device)
        input_shape = self.input_shape.to(model_output.device)
        cell_shape = input_shape / cell_map_shape
        x_norm_rel, y_norm_rel, w_norm_sqrt, h_norm_sqrt, pred_conf, pred_cls_conf = self.parse_output(model_output)

        x_rel = x_norm_rel * cell_shape[0]
        y_rel = y_norm_rel * cell_shape[1]
        w = w_norm_sqrt ** 2 * input_shape[0]
        h = h_norm_sqrt ** 2 * input_shape[1]
        cell_top_left = yolo_grid(input_shape, cell_map_shape)
        bbox_top_left_corner = cell_top_left[None, :, :, None, :] + torch.stack([x_rel, y_rel], dim = -1)

        if threshold is None:
            return torch.cat(
                [bbox_top_left_corner, w.unsqueeze(-1), h.unsqueeze(-1), pred_conf.unsqueeze(-1)], dim = -1
                )

        detected_bbox_idx = torch.nonzero(torch.gt(pred_conf, threshold)).split(1, dim = -1)
        batch_idx = detected_bbox_idx[0]
        if batch_idx.shape[0] == 0:
            return torch.zeros([0, 7])

        detected_top_left_corner = bbox_top_left_corner[detected_bbox_idx].squeeze(1)
        detected_h = h[detected_bbox_idx]
        detected_w = w[detected_bbox_idx]
        pred_conf = pred_conf[detected_bbox_idx]

        pred_cls = torch.argmax(pred_cls_conf[detected_bbox_idx[:-1]], dim = -1)
        pred_cls_conf = pred_cls_conf[detected_bbox_idx[:-1]].squeeze(1)
        pred_cls_conf = pred_cls_conf[torch.arange(pred_cls.shape[0]), pred_cls.squeeze(-1)]

        # Convert from x, y to u, v
        det_bbox = torch.cat(
            [batch_idx.float(), detected_top_left_corner[:, 0, None].float(),
             detected_top_left_corner[:, 1, None].float(), detected_w.float(), detected_h.float(),
             pred_cls.float(), pred_cls_conf[:, None].float(), pred_conf], dim = -1
            )

        det_bbox[:, 1:5] = crop_to_frame(det_bbox[:, 1:5], image_shape = input_shape)
        return det_bbox

    def detect_nms(self, model_outputs: torch.Tensor, threshold: float = 0.6, nms_iou: float = 0.6):
        detected_bbox = self.detect(model_outputs, threshold = threshold)
        return non_max_suppression(detected_bbox, iou = nms_iou)