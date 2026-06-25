class ImageRegressionPrompts: 
    def __init__(self):
        self.prompts = [
            # General Refinement 
            '"""Refine the model to minimize Mean Squared Error (MSE) for a regression task, without altering the I/O layers."""',
            '"""Re-balance the architecture for a better trade-off between parameter count and prediction accuracy."""',

            # Width and Channel Adjustments
            '"""Widen the convolutional base by increasing the output channels of all `nn.Conv2d` layers by 25%."""',
            '"""Make the regressor head more powerful by increasing its hidden layer neurons from 512 to 1024."""',
            '"""Decrease the networks width by halving the output channels of the `conv1` and `conv2` blocks."""',

            #  Component & Parameter Tuning
            '"""Replace the `nn.ReLU` activation functions with `nn.LeakyReLU(negative_slope=0.01)` to handle negative values."""',
            '"""Remove the `nn.Dropout` layer from the regressor head, as it can sometimes harm regression performance."""',
            '"""Change the kernel size in the `conv1` layer to 7x7 for a larger initial receptive field, adjusting padding accordingly."""',
            '"""Replace the `nn.MaxPool2d` layers with a strided convolution with a kernel size of 2 and stride of 2."""',
            
            # Layer Addition (Normalization & Regularization)
            '"""Insert a `nn.BatchNorm2d` layer within each convolutional block to improve gradient flow."""',
            '"""Add a `nn.LayerNorm` layer to the regressor head for stable activation statistics."""',
            '"""Integrate a simple spatial attention mechanism before the `adaptive_pool` layer."""',
            '"""Incorporate an additional `nn.Linear` layer with 256 neurons into the regressor head."""',

            # Structural & Final Layer Changes 
            '"""Introduce a single residual connection that adds the output of `conv1` to the output of `conv3`. Ensure the residual connection has matching spatial dimensions and channels."""',
            '"""Replace the standard convolution in the `conv3` block with a depthwise separable convolution to make the model lighter."""',
            '"""Add a `nn.Sigmoid` activation to the final output layer to ensure all predictions are between 0 and 1."""',
            '"""Add a `nn.Tanh` activation to the final output layer to constrain predictions between -1 and 1."""',
            '"""Add a `nn.Hardtanh` layer to the final output to clip predictions within a specific min-max range."""',
            '"""Refactor the regressor head into its own `nn.Module` class for better encapsulation. Keep the main ImageRegressionCNN class as the primary network class."""',
            '"""Make the model slightly deeper by adding a small convolutional block with 1x1 convolutions after `conv3`."""'
        ]

    def get_prompts(self):
        return self.prompts

def get_image_regression_full_prompt(instruction_prompt, base_net_code):
    full_prompt = (
            "You are an expert PyTorch programmer specializing in multi layer Convolutional Neural Networks for an image regression task."
            "The network should take a 3-channel image as input and output 3 prediction values."
            "Your task is to modify the following ImageRegressionCNN class according to the specified instruction. "
            f"The required modification is: {instruction_prompt}\n\n"
            "CRITICAL: Provide ONLY the complete, new `class ImageRegressionCNN(nn.Module):` block as your output. "
            "Do NOT include any markdown formatting, backticks, or code block markers. "
            "Start directly with the class definition and end with the closing brace.\n\n"
            "### Original Code:\n"
            "```python\n"
            f"{base_net_code}\n"
            "```\n\n"
            "### New Modified Code:\n"
            "Provide the complete class definition here (no markdown formatting):\n"
        )
    return full_prompt