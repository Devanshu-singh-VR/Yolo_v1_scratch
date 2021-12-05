# if IOU > 0.5 good
# if IOU > 0.7 pretty good
# if IOU > 0.9 excellent

import torch

def intersection_over_union(pred, label, box_format="midpoint"):
    # boxes will have the shape of (N, 4)

    if box_format == 'midpoint':
        box1_x1 = pred[..., 0:1] - pred[..., 2:3] / 2
        box1_y1 = pred[..., 1:2] - pred[..., 3:4] / 2
        box1_x2 = pred[..., 2:3] + pred[..., 2:3] / 2
        box1_y2 = pred[..., 3:4] + pred[..., 3:4] / 2

        box2_x1 = label[..., 0:1] - label[..., 2:3] / 2
        box2_y1 = label[..., 1:2] - label[..., 3:4] / 2
        box2_x2 = label[..., 2:3] + label[..., 2:3] / 2
        box2_y2 = label[..., 3:4] + label[..., 3:4] / 2

    elif box_format == 'corner':
        box1_x1 = pred[..., 0:1]
        box1_y1 = pred[..., 1:2]
        box1_x2 = pred[..., 2:3]
        box1_y2 = pred[..., 3:4]

        box2_x1 = label[..., 0:1]
        box2_y1 = label[..., 1:2]
        box2_x2 = label[..., 2:3]
        box2_y2 = label[..., 3:4]
    # here if we do just label[..., 3] it will give
    # the shape (N), and if we slice the data label[..., 3:4]
    # then we cab keep the shape (N, 1)

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(min, max) if we do min=0 then all the value
    # less than 0 will be converted to 0.
    # hence if x2-x1 or y2-y1 is negative of o means no Intersecting
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # abs because if the are is negative
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection + 1e-9

    return intersection/union

if __name__ == '__main__':
    pred = torch.rand((10, 4))
    label = torch.rand((10, 4))

    pred[:, 0] = 2
    pred[:, 1] = 2
    pred[:, 2] = 4
    pred[:, 3] = 4
    label[:, 0] = 3
    label[:, 1] = 3
    label[:, 2] = 5
    label[:, 3] = 5

    print(intersection_over_union(pred, label, box_format='corner'))
