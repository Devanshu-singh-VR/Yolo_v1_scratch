import torch
import os
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, csv_path, img_path, label_path,
                 S=7, B=2, C=20, transforms=None):
        super(VOCDataset, self).__init__()
        self.annotations = pd.read_csv(csv_path)
        self.img_path = img_path
        self.label_path = label_path
        self.transform = transforms
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        label_path = os.path.join(self.label_path, self.annotations.iloc[item, 1])
        boxes = []
        # extracting the labels
        labels = open(label_path, 'r')
        for box in labels:
            class_label, x, y, w, h = box.split()

            boxes.append([class_label, x, y, w, h])

        # extracting the images
        image = os.path.join(self.img_path, self.annotations.iloc[item, 0])
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + self.B*5)) # for the label
        for box in boxes:
            class_label, x, y, w, h = box
            # make item to int and float
            class_label = int(class_label)
            x, y, w, h = float(x), float(y), float(w), float(h)

            # relative to the cell
            i, j = int(self.S * y), int(self.S * x) # i, j where the bbox should be in
            x_cell, y_cell = self.S*x - j, self.S*y - i # doing reverse of that will give the original coordinates
            w_cell, h_cell = (
                w * self.S,
                h * self.S
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                label_matrix[i, j, 21:25] = torch.tensor(
                    [x_cell, y_cell, w_cell, h_cell])
                label_matrix[i, j, class_label] = 1

        return image, label_matrix

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((488, 488)),
        transforms.ToTensor()
    ])
    dataset = VOCDataset('8examples.csv', 'images',
                         'labels', transforms=transform)

    for a, b in dataset:
        print(a.shape, b.shape)

