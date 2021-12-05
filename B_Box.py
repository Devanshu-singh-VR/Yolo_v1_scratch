import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def cell_to_box(score, S):
    score = score.to('cpu')
    batch_size = score.shape[0]
    prediction = score.reshape(batch_size, S, S, 30)

    box1 = prediction[..., 21:25]
    box2 = prediction[..., 26:30]

    max_value, best_score = torch.max(torch.cat(
        (prediction[..., 20:21].unsqueeze(0), prediction[..., 25:26].unsqueeze(0)),
        dim=0), dim=0)
    # max_value, best_score .shape(batch, 7, 7, 1)

    best_box = box1 * (1 - best_score) + box2 * (best_score)
    # best_box.shape(batch, S, S, 4)

    # that is the reverse of what we did in the dataset class
    cells = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1/S * (best_box[..., 0:1] + cells)
    y = 1/S * (best_box[..., 1:2] + cells.permute(0, 2, 1, 3))
    w_h = 1/S * best_box[..., 2:]

    cell_box = torch.cat((x, y, w_h), dim=-1)
    # cell_box.shape(batch, S, S, 4)

    class_label = prediction[..., :20].argmax(-1).unsqueeze(-1)
    # class_label.shape(batch, S, S, 1)

    final_matrix = torch.cat((class_label, max_value, cell_box), dim=-1)
    # final_matrix.shape(batch, S, S, 6)
    # we convert it into this format because we want to pass them through NMS class baby

    return final_matrix

def box_to_boxes(score, S):
    batch_size = score.shape[0]
    matrix = cell_to_box(score, S).reshape(score.shape[0], S*S, -1)
    boxes = []

    for batch in range(batch_size):
        box = []
        for i in range(S*S):
            box.append([b.item() for b in matrix[batch, i, :]])
        boxes.append(box)

    return boxes

# the data will be taken from the train_loader where
# image size will be (480, 480, 3)
# we can do on random size data but it require to write some code
# and i can't do it because my gradients got vanished
def plot_img(image, boxes):
    image = np.array(image)
    width, height = image.shape[0], image.shape[1]
    fig, ax = plt.subplots(1)

    ax.imshow(image)

    for box in boxes:
        box = box[2:]

        mid_x = box[0] * width
        mid_y = box[1] * height

        box[2] = box[2] * width
        box[3] = box[3] * height

        box[0] = (2*mid_x - box[2])/2
        box[1] = (2*mid_y - box[3])/2

        rect = patches.Rectangle((box[0], box[1]), box[2], box[3],
                                 linewidth=1, edgecolor='r', facecolor="none")

        ax.add_patch(rect)

    plt.show()


