# Image Deblurring App
**"Image Deblurring Using Deep Learning"** is a cutting-edge project that leverages a powerful stack of technologies to address the challenge of image debarring. Built with **Python** as the primary programming language and **Visual Studio Code** as the integrated development environment, this project offers a user-friendly **Tkinter GUI** for seamless interaction. Under the hood, it employs **PyTorch** as the deep learning framework and utilises image processing libraries like **OpenCV, Scikit-image, and Pillow** to enhance and restore blurred images with remarkable precision.

## Technology Stacks
- **Programming Language**: Python
- **IDE**: Visual Studio Code
- **User Interface**: Tkinter GUI
- **Deep Learning Framework**: Pytorch
- **Image Processing Libraries**: OpenCV, Scikit-image, and Pillow

## Getting Started
1. Clone the repository into your local system
```bash
git clone https://github.com/aniru-dh21/Image-Deblurring-App.git
```
2. Open terminal and change the working directory to the following:
```bash
cd ./src 
```
3. To run the application first you have to train the model, so first run the following with the command;
```py
python deblur.py
```
4. After completion of execution, you can now test the application using the following command:
```py
python test.py
```
5. By running above command, Tkinter GUI window will open where you can pass the any blurred image to get an output of deblurred image.

### Code Snippet
```py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(128, 128, kernel_size=5, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x += residual

        return x
```
The code defines a convolutional neural network (CNN) class with three convolutional layers and batch normalization. It takes an input tensor 'x', applies convolutions with activation functions and batch normalization, and adds the original input as a residual connection.
```py
class SimpleAE(nn.Module):
    def __init__(self):
        super(SimpleAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.BatchNorm2d(3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x
```
The code defines a simple autoencoder neural network (AE) with an encoder and decoder. The encoder consists of convolutional layers with batch normalization and ReLU activation, which reduce the input's dimensionality. The decoder uses transpose convolutions, batch normalization, and a sigmoid activation to reconstruct the input from the encoded representation.
```py
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
```
This code defines the training function for the model using a given dataloadder. It iterates through the training data, computes the loss between the model's predictions and the ground truth, performs backpropagation to update model optimizing the model during the training process.
