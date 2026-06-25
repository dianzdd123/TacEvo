"""
Prompts for neural architecture generation for different base architectures
- ResNet-18
- 4 layer CNN for single label classification
- 4 layer CNN for image regression
"""

from .image_regression_prompts import ImageRegressionPrompts, get_image_regression_full_prompt
from .single_label_prompts import SingleLabelPrompts, get_single_label_full_prompt
from .resnet_prompts import ResNetPrompts, get_resnet_full_prompt

__all__ = [
    'ImageRegressionPrompts',
    'get_image_regression_full_prompt',
    'SingleLabelPrompts',
    'get_single_label_full_prompt',
    'ResNetPrompts',
    'get_resnet_full_prompt'
]
