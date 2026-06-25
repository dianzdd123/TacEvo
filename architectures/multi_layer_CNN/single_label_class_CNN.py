import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleLabelClassCNN(nn.Module):
    """
    A Convolutional Neural Network for single-label image classification.
    Architecture: 3 conv layers, 2 FC layers.
    """
    def __init__(self, num_classes):
        super(SingleLabelClassCNN, self).__init__()

        #  Convolutional Block 1 
        # Takes a 3-channel image (RGB) and applies convolution, activation, and pooling.
        # Input: (batch_size, 3, H, W) -> Output: (batch_size, 64, H/2, W/2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #  Convolutional Block 2 
        # Increases the feature depth.
        # Input: (batch_size, 64, H/2, W/2) -> Output: (batch_size, 128, H/4, W/4)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #  Convolutional Block 3 
        # Increases feature depth.
        # Input: (batch_size, 128, H/4, W/4) -> Output: (batch_size, 256, H/8, W/8)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #  Adaptive Pooling and Classifier Head 
        # Adaptive pooling creates a fixed-size output
        # making the model flexible to different input image sizes.
        self.adaptive_pool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        # The classifier part consists of two fully connected layers.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512), # 256 channels * 7x7 pooled output
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes) # The final output layer
        )

    def forward(self, x):
        """
        Defines the forward pass of the model
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        # Return raw logits for CrossEntropyLoss ( combines LogSoftmax + NLLLoss internally)
        return x