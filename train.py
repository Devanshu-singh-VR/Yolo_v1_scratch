# the GPU memory will get full after running the code
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from torchvision.transforms import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolo
from dataset import VOCDataset
from loss import Loss
from NMS import non_max_suppression
from B_Box import box_to_boxes, plot_img

seed = 123
torch.manual_seed(seed)

# Hyperparameters
learning_rate = 2e-5
device = "cuda" if torch.cuda.is_available() else "cpu"
Batch_size = 1
weight_decay = 0
num_epochs = 5
workers = 5
pin_memory = True

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

def train_fn(train_loader, model, optimizer, loss_fn):
    mean_loss = []

    for batch, (train, label) in enumerate(train_loader):
        train = train.to(device)
        label = label.to(device)
        score = model(train)

        loss = loss_fn(score, label)
        mean_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('The loss ', sum(mean_loss)/len(mean_loss))

def main():
    model = Yolo(in_channels=3, split_size=7, num_boxes=2, num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = Loss()


    train_dataset = VOCDataset('8examples.csv', 'images',
                               'labels', transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=Batch_size,
                              num_workers=workers, pin_memory=pin_memory,
                              shuffle=True)

    # train the model epochs one by one.

    for epoch in range(num_epochs):
        train_fn(train_loader, model, optimizer, loss_fn)

    for x, y in train_loader:
        x = x.to(device)
        for idx in range(1):
            bboxes = box_to_boxes(model(x), 7)
            bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, prob_threshold=0.5, box_format="midpoint")
            plot_img(x[idx].permute(1, 2, 0).to("cpu"), bboxes)

if __name__ == '__main__':
    main()





