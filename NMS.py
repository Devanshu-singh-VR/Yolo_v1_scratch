import torch
from IOU import intersection_over_union
# if the bboxes probability if less then some threshold (like 0.2) then remove it
# while bboxes:
#       take largest bboxes prediction box
#       remove the box in boxes if its iou with largest box is > iou threshold
#       why we select which has less iou threshold ??? -> you will see in mean Avj precision

def non_max_suppression(bboxes, iou_threshold,
                        prob_threshold, box_format):
    # let the bboxes = [[1, 0.9, x1, x2, x3, x4]] class, predicted_val_of_class, bboxes

    assert type(bboxes) == list # if the it is not in the format of list
                                # it will generate an error.

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    # if you didn't get the format in list
    # [box
    #  for box in bboxes
    #  if box[1] > some value]
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse=True) # sorting according to prob
    bboxes_after_nmr = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if (box[0] != chosen_box[0])
            or (intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format) < iou_threshold)
        ]

        bboxes_after_nmr.append(chosen_box)

    return bboxes_after_nmr


if __name__ == '__main__':
    a = [[1, 0.9, 2, 2, 4, 4], [1, 0.8, 3, 3, 5, 5], [1, 0.2, 2.5, 2.5, 4, 4], [2, 0.3, 3, 3, 4, 4]]
    print(non_max_suppression(a, 0.2, 0.2, box_format='corner'))
