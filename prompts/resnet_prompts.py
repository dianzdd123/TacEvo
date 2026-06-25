class ResNetPrompts:
    def __init__(self):
        self.prompts = [
        # Block and Layer Manipulation
        # '"""Add a new BasicBlock to self.layer3 to deepen the network."""',  # HIGH IMPACT - COMMENTED OUT
        '"""Remove one BasicBlock from self.layer2 to make the network shallower."""',
        # '"""Duplicate self.layer2 and insert it after the original self.layer2."""',  # HIGH IMPACT - COMMENTED OUT
        '"""Replace the self.layer4 with a new layer that has 3 BasicBlocks."""',
        # '"""Increase the number of blocks in all layers by one."""',  # HIGH IMPACT - COMMENTED OUT

        # Channel and Dimension Adjustments 
        '"""Double the number of output channels in self.layer3."""',
        '"""Halve the number of output channels in self.layer2."""',
        '"""Change the number of output channels for all layers to [64, 128, 256, 512] respectively."""',

        # Introducing New Layer Types
        '"""Insert a squeeze-and-excitation block within each BasicBlock in self.layer3."""',
        '"""Add an attention mechanism after self.layer4 and before the self.avgpool layer."""',
        '"""Replace the nn.MaxPool2d with an average pooling layer with the same kernel size and stride."""',

        # BasicBlock Structure Changes
        '"""Modify the BasicBlock to include a dropout layer with p=0.2 after the first ReLU."""',
        '"""In the BasicBlock, change the kernel size of the first convolutional layer from 3 to 5 and adjust padding accordingly."""',
        # '"""In the BasicBlock, add a third convolutional layer to create a bottleneck structure."""',  # HIGH IMPACT 
        '"""Replace the nn.ReLU with nn.GELU as the activation function throughout the BasicBlock."""',
        '"""Change the order of operations in the BasicBlock to Conv -> ReLU -> BN."""',

        #  Downsample and Skip Connection Modifications 
        '"""Modify the downsample block to use average pooling instead of a strided convolution."""',
        '"""Remove the downsample functionality from the network."""',

        # General Improvements
        '"""Improve the performance of this ResNet model by making it more parameter-efficient."""',
        '"""Refactor the ResNet to be more computationally efficient without significantly sacrificing accuracy."""',
        '"""Introduce a mechanism to improve gradient flow through the network."""',
        '"""Modify the network to be more robust to overfitting."""',
        # '"""Transform this ResNet-18 into a ResNet-34 architecture by adjusting the number of blocks in each layer."""',  # HIGH IMPACT
    ]
        
    def get_prompts(self):
        return self.prompts
    
def get_resnet_full_prompt(instruction_prompt, base_net_code):
    full_prompt = (
            "You are an expert PyTorch programmer specializing in ResNet architectures. "
            "Your task is to modify the following ResNet class according to the specified instruction. "
            f"The required modification is: {instruction_prompt}\n\n"
            "CRITICAL: Provide ONLY the complete modified ResNet class code. "
            "Do NOT include any markdown formatting, backticks, or code block markers. "
            "Start directly with the class definition and end with the closing brace.\n\n"
            "IMPORTANT REQUIREMENTS:\n"
            "1. Maintain the ResNet architecture structure with proper skip connections\n"
            "2. Keep the BasicBlock class intact unless specifically asked to modify it\n"
            "3. Ensure the forward method properly handles the residual connections\n"
            "4. Preserve the layer naming convention (layer1, layer2, layer3, layer4)\n\n"
            "### Original ResNet Code:\n"
            "```python\n"
            f"{base_net_code}\n"
            "```\n\n"
            "### Modified ResNet Code:\n"
            "Provide the complete class definition here (no markdown formatting):\n"
        )
    return full_prompt