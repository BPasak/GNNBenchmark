import torch
from torch.nn.functional import cross_entropy, l1_loss

from External.EGSST_PAPER.detector.rtdetr_head.rtdetr_box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from External.EGSST_PAPER.detector.rtdetr_head.rtdetr_criterion import accuracy


## From the EGSST RTDETR Head Implementation
def _get_src_permutation_idx(indices):
    # permute predictions following indices
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

## From the EGSST RTDETR Head Implementation
def _get_tgt_permutation_idx(indices):
    # permute targets following indices
    batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
    tgt_idx = torch.cat([tgt for (_, tgt) in indices])
    return batch_idx, tgt_idx


## From the EGSST RTDETR Head Implementation
def loss_boxes(outputs, targets, indices, num_boxes):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    assert 'pred_boxes' in outputs
    idx = _get_src_permutation_idx(indices)
    src_boxes = outputs['pred_boxes'][idx]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

    losses = {}

    loss_bbox = l1_loss(src_boxes, target_boxes, reduction='none')
    losses['loss_bbox'] = loss_bbox.sum() / num_boxes

    loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
    losses['loss_giou'] = loss_giou.sum() / num_boxes
    return losses

def loss_labels(outputs, targets, indices, num_classes):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    assert 'pred_logits' in outputs
    src_logits = outputs['pred_logits']

    idx = _get_src_permutation_idx(indices)
    target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
    target_classes = torch.full(src_logits.shape[:2], num_classes,
                                dtype=torch.int64, device=src_logits.device)
    target_classes[idx] = target_classes_o

    loss_ce = cross_entropy(src_logits.transpose(1, 2), target_classes)
    losses = {'loss_ce': loss_ce}
    return losses