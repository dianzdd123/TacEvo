"Training functions for each base architecture"

# Version information
__version__ = "1.0.0"

# Import commonly used files at package level
from .multi_layer_CNN import single_label_CNN_train_utils, image_regression_CNN_train_utils
from .resnet import resnet_training_utils 

__all__ = [
    'single_label_CNN_train_utils',
    'image_regression_CNN_train_utils', 
    'resnet_training_utils'
]
