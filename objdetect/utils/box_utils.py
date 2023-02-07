import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x: torch.Tensor):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x: torch.Tensor):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def box_clamp_01(x):
    x = torch.clamp(x, min=0, max=1)
    return x


def degenerate_mask(boxes, needs_format=False):
    if needs_format:
        boxes = box_cxcywh_to_xyxy(boxes)
    return (boxes[:, 2:] <= boxes[:, :2]).all(-1)


def box_iou(boxes1, boxes2, needs_format=False):
    if needs_format:
        boxes1 = box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_cxcywh_to_xyxy(boxes2)

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def positive_negative(
    boxes1, boxes2, threshold=0.5
):  # boxes1 is the ground truth, boxes2 is the prediction, Nx4 both
    boxes1 = box_cxcywh_to_xyxy(boxes1)
    boxes2 = box_cxcywh_to_xyxy(boxes2)
    mask = degenerate_mask(boxes1) or degenerate_mask(boxes2)
    iou, union = box_iou(boxes1[mask], boxes2[mask])
    N = boxes1.shape[0]
    ret = torch.zeros(N, dtype=torch.bool)
    ret[mask] = iou.diagonal() > threshold
    return ret


def generalized_box_iou(boxes1, boxes2, needs_format=False):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in cxchwh format. (fixed)
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    if needs_format:
        boxes1 = box_cxcywh_to_xyxy(boxes1)
        boxes2 = box_cxcywh_to_xyxy(boxes2)
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area
