class SingleLabelPrompts: 
    def __init__(self):
            self.prompts = [
        # General Refinement
        '"""Refine the model to achieve higher classification accuracy while preserving the I/O structure."""',
        '"""Optimize the model for faster inference speed by making minor architectural adjustments."""',

        # Width and Channel Adjustments
        '"""Widen the network by increasing the output channels of the `conv2` block by 25%."""',
        '"""Make the network slightly narrower by decreasing the number of neurons in the hidden fully connected layer to 384."""',
        '"""Increase the complexity of the classifier head by adding a second hidden layer with 256 neurons."""',

        # Component & Parameter Tuning
        '"""Replace the `nn.ReLU` activation functions with `nn.GELU` for potentially smoother gradients."""',
        '"""Adjust the dropout probability in the classifier head to 0.6 for stronger regularization."""',
        '"""Change the kernel size in the `conv3` block to 5x5, adjusting padding to maintain the feature map size."""',
        '"""Replace the `nn.MaxPool2d` layers with `nn.AvgPool2d` for a different pooling strategy."""',

        # Layer Addition (Normalization & Regularization) 
        '"""Insert a `nn.BatchNorm2d` layer after each `nn.Conv2d` layer to stabilize training. Use LayerNorm instead of BatchNorm in the classifier to avoid batch size issues."""',
        '"""Add a `nn.LayerNorm` layer to the classifier head just before the `nn.ReLU` activation."""',
        '"""Incorporate an additional `nn.Dropout` layer after the first `nn.Linear` layer in the classifier."""',

        # Structural & Final Layer Changes 
        '"""Introduce a single residual connection that skips over the `conv2` block."""',
        '"""Change the final activation to a standard `nn.Softmax(dim=1)` for direct probability outputs."""',
        '"""Replace `nn.AdaptiveAvgPool2d` with a combination of adaptive average and max pooling to capture more diverse features."""',
        '"""Refactor the three `nn.Sequential` convolutional blocks into a single custom `nn.Module` for better code organization."""',
        '"""Make the classifier head more lightweight by reducing its hidden layer size and removing dropout."""'
    ]

    def get_prompts(self):
        return self.prompts


def get_single_label_full_prompt(instruction_prompt, base_net_code):
    full_prompt = (
                "You are an expert PyTorch programmer specializing in multi layer Convolutional Neural Networks for a single-label classification task."
                "Your task is to modify the following SingleLabelClassCNN class according to the specified instruction. "
                f"The required modification is: {instruction_prompt}\n\n"
                "CRITICAL: Provide ONLY the complete, new `class SingleLabelClassCNN(nn.Module)` block as your output. "
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