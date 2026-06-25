import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageRegressionCNN(nn.Module):
    """
    A Convolutional Neural Network for an image regression task.
    This model takes an image as input and outputs 3 continuous values.
    Architecture: 3 conv layers, 2 FC layers.
    """
    def __init__(self, output_size=3):
        super(ImageRegressionCNN, self).__init__()

        # Convolutional Blocks 
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #  Adaptive Pooling and Classifier Head 
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # The classifier (or regressor in this case) part.
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.regressor(x)
        return x