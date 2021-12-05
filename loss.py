import torch
import torch.nn as nn
from IOU import intersection_over_union

class Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        predictions = predictions.reshape(-1, self.S, self.S, (self.C + self.B*5))
        # prediction.shape(Batch, S, S, 30)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_max_val, bestbox = torch.max(ious, dim=0) # will be (2, 62), where max value, argmax of the max value will return
        exists_box = target[..., 20].unsqueeze(3) # it will tell there is object or not
        # exists_box.shape(batch, S, S, 1)

        '''FOR BOX Coordinates'''
        # for the mid points
        box_predictions_xy = exists_box * (
            (
                (1 - bestbox) * predictions[..., 21:23]
                + bestbox * predictions[..., 26:28]
            )
        )

        box_target_xy = exists_box * target[..., 21:23]

        # for the width and height
        """ The loss i had for width and height
        box_predictions_wh = exists_box * (
            (
                    (1 - bestbox) * predictions[..., 23:25]
                    + bestbox * predictions[..., 28:30]
            )
        )

        box_target_wh = exists_box * target[..., 23:25]
        

        # filtering it according to the paper
        box_predictions_wh_filter = torch.sign(box_predictions_wh[..., 0:2]) * torch.sqrt(
            torch.abs(box_predictions_wh[..., 0:2]) + 1e-6)

        box_target_wh_filter = torch.sqrt(box_target_wh[..., 0:2])
        """

        # for the width and height
        # filtering it according to the paper width and height
        box_predictions_wh_filter = torch.sign(exists_box * (
            (
                    (1 - bestbox) * predictions[..., 23:25]
                    + bestbox * predictions[..., 28:30]
            )
        )) * torch.sqrt(
            torch.abs(exists_box * (
                (
                        (1 - bestbox) * predictions[..., 23:25]
                        + bestbox * predictions[..., 28:30]
                )
            )) + 1e-6)

        box_target_wh_filter = torch.sqrt(exists_box * target[..., 23:25])

        # I removed it because it is an inplace operation
        # when loss backward computes gradient it need the previous value (which is changed by b[0] = something)
        # it creates error.
        # for more information: http://www.yongfengli.tk/2018/04/13/inplace-operation-in-pytorch.html
        '''
        # loss for the width and height is SUM( (root(h) - root(h^))^2 + (root(w) - root(w^))^2 )
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4]) + 1e-6) # width and height may have negative values in starting
        # to avoide the error we has use absolute value and using sign because after sqrt the sign will diminish.

        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4]) # target will not have negative (ofcourse).
        '''

        # here is the loss value (N, S, S, 4) -> (N*S*S, 4)
        box_loss_xy = self.mse(
            torch.flatten(box_predictions_xy, end_dim=-2),
            torch.flatten(box_target_xy, end_dim=-2)
        )

        box_loss_wh = self.mse(
            torch.flatten(box_predictions_wh_filter, end_dim=-2),
            torch.flatten(box_target_wh_filter, end_dim=-2)
        )

        '''OBJECT Loss'''

        pred_box = (
            (1 - bestbox) * predictions[..., 20:21]
            + bestbox * predictions[..., 25:26]
        )

        # obj loss (N*S*S)
        object_loss = self.mse(
            torch.flatten(pred_box * exists_box),
            torch.flatten(target[..., 20:21] * exists_box)
        )

        '''NO OBJECT Loss'''
        # (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        ) + self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        '''Class Loss'''
        # (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(predictions[..., :20], end_dim=-2),
            torch.flatten(target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss_xy
            + self.lambda_coord * box_loss_wh
            + object_loss
            + self.lambda_no_obj * no_object_loss
            + class_loss
        )

        return loss

if __name__ == '__main__':
    loss = Loss()

    pred = torch.ones((10, 7, 7, 30))
    targ = torch.ones((10, 7, 7, 25)) * 4
    print(loss(pred, targ))
