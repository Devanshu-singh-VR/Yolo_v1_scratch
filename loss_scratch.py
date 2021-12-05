import torch
import torch.nn as nn
from IOU import intersection_over_union

# loss created by me from scratch bass ese hi :)
class Loss(nn.Module):
    def __init__(self, S=7, C=20, B=2):
        super(Loss, self).__init__()
        self.S = S
        self.C = C
        self.B = B
        self.mse = nn.MSELoss(reduction='sum')
        self.lambda_no_obj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        # prediction.shape(batch, S*S*( C+ B*5 ))
        prediction = prediction.reshape(prediction.shape[0], self.S, self.S, self.C + self.B*5)

        iou1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25]).unsqueeze(0)
        iou2 = intersection_over_union(prediction[..., 26:30], target[..., 21:25]).unsqueeze(0)
        # shape of IOUS (1, batch, S, S, 4)
        max_val, argmax = torch.max(torch.cat((iou1, iou2), dim=0), dim=0)
        # argmax.shape(batch, S, S, 4)
        exist_object = target[..., 20:21]

        '''loss for exist objects boxes'''

        boxes_xy = exist_object * (
            (
                (1 - argmax) * prediction[..., 21:23]
                + argmax * prediction[..., 26:28]
            )
        )
        # boxes_xy.shape(batch, S, S, 2)

        boxes_wh = torch.sign(
            exist_object * (
                prediction[..., 23:25] * (1-argmax)
                + prediction[..., 28:30] * argmax
            ) * torch.sqrt(
                torch.abs(exist_object * (
                    prediction[..., 23:25] * (1-argmax)
                    + prediction[..., 28:30] * argmax
                    )
                )
                + 1e-6)
        )

        loss_box_xy = self.mse(
            torch.flatten(boxes_xy, end_dim=-2), # you can also reshape here rather then flatten
            torch.flatten(target[..., 21:23], end_dim=-2)
        )

        loss_box_wh = self.mse(
            torch.flatten(boxes_wh, end_dim=-2),  # you can also reshape here rather then flatten
            torch.flatten(target[..., 23:25], end_dim=-2)
        )

        '''loss for classes'''

        loss_class = self.mse(
            torch.flatten(prediction[..., :20], end_dim=-2),
            torch.flatten(target[..., :20], end_dim=-2)
        )

        '''object loss'''

        obj_pred = exist_object * (
            prediction[..., 20:21] * (1 - argmax)
            + prediction[..., 25:26] * argmax
        )

        loss_obj = self.mse(
            torch.flatten(obj_pred),
            torch.flatten(target[..., 20:21])
        )

        '''None object loss'''

        loss_no_obj = self.mse(
            torch.flatten((1-exist_object) * prediction[..., 20:21]),
            torch.flatten((1-exist_object * target[..., 20:21]))
        ) + self.mse(
            torch.flatten((1-exist_object) * prediction[..., 25:26]),
            torch.flatten((1-exist_object * target[..., 20:21]))
        )


        loss = (
            self.lambda_coord * loss_box_xy
            + self.lambda_coord * loss_box_wh
            + self.lambda_no_obj * loss_no_obj
            + loss_obj
            + loss_class
        )

        return loss

if __name__ == '__main__':
    loss = Loss()

    pred = torch.ones((10, 7, 7, 30))
    targ = torch.ones((10, 7, 7, 25)) * 4
    print(loss(pred, targ))