"""
Authors: V. Anirudh, P. Bhargava Rao, B. Shiva Shankar, A. Abinav Reddy
Title: Image Deblurring Using Deep Directory
Models: Autoencoders + Convolutional Neural Network
Dataset: RealBlur Dataset
"""

# Importing the required dependencies
import os
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time 
import argparse
import models
import yaml

# Importing the required packages
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split

# Read parameters
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

epochs = config.get("epochs", 60) 
path = config.get("path", "C:/Users/ANIRUDH/OneDrive/Desktop/Image Deblurring App")

# helper functions
image_dir = path + '/outputs/saved_images'
os.makedirs(image_dir, exist_ok=True)

# Functions for viewing the image in size of 224 x 224
def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 224, 224)
    save_image(img, name)

# To check for availability of GPU memory on the machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

# Batch Size of images
batch_size = 3

# Directories for training images and CNN, Autoencoders models
gauss_blur = os.listdir(path + '/input/gaussian_blurred')
gauss_blur.sort()
sharp = os.listdir(path + '/input/sharp')
sharp.sort()

# This is used for checking that whether the blur image is regarding to the corresponding sharp image.
x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])

y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])

print(x_blur[10])
print(y_sharp[10])

# Train and Test split with 20% to be used as test dataset
(x_train, x_val, y_train, y_val) = train_test_split(x_blur, y_sharp, test_size=0.20)

print(len(x_train))
print(len(x_val))

# define transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Deblurring transformations
class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms

    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        blur_image = cv2.imread(path + f"/input/gaussian_blurred/{self.X[i]}")

        if self.transforms:
            blur_image = self.transforms(blur_image)

        if self.y is not None:
            sharp_image = cv2.imread(path + f"/input/sharp/{self.y[i]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image

# Used to load and generate the image into tensors and arrays of size 224 x 224. 
train_data = DeblurDataset(x_train, y_train, transform)
val_data = DeblurDataset(x_val, y_val, transform)

trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Models to be used which is CNN and Autoencoders
model = models.SimpleAE().to(device)
print(model)

# the loss function
criterion = nn.MSELoss()

# the optimizer and Learning rate scheduler
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode = 'min',
    patience=5,
    factor=0.1,
    verbose=True
)

# optimizer function
def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)

        # backpropagation
        loss.backward()

        # update the parameters
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")

    return train_loss

# the training function
def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()

            # based on the epoch number used for training and evaluation
            if epoch == 0 and i == (len(val_data)/dataloader.batch_size)-1:
                save_decoded_image(sharp_image.cpu(), name=path + f"/outputs/saved_images/sharp[epoch].jpg")
                save_decoded_image(blur_image.cpu(), name=path + f"/outputs/saved_images/blur[epoch].jpg")

        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")

        save_decoded_image(outputs.cpu().data, name=path + f"/outputs/saved_images/val_deblurred[epoch].jpg")

        return val_loss

# Evaluating and Plotting loss function
train_loss = []
val_loss = []
start = time.time()
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(val_epoch_loss)
end = time.time()

print(f"Took {((end-start)/60):.3f} minutes to train")

# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(path + '/outputs/loss.png')
plt.show()

# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), path + '/outputs/model.pth')
