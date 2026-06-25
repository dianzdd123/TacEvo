import torch
from fvcore.nn import FlopCountAnalysis
import torch.nn as nn
import numpy as np
from conf.config import Config

cfg = Config()


# Efficiency Ratio Descriptor for QD Optimization

# Uses the output of the flops and param_count functions in utils.py as inputs
def efficiency_ratio(model, cfg: Config):
    """
    Calculate the efficiency ratio of a model.
    Uses the output of the flops and param_count functions in utils.py as inputs
    """
    # calculate width depth ratio
    width_depth_ratio = get_network_width_depth_ratio(model)
    
    # calculate flops
    flops = calculate_flops_robust(model, cfg)
    
    # calculate param count
    param_count = count_parameters(model)
    
    # Check for valid values before calculation
    if width_depth_ratio == 0 or param_count == 0 or flops == 0:
        print(f"Warning: Invalid values for efficiency calculation")
        print(f"param_count: {param_count}, flops: {flops}, depth: {width_depth_ratio}")
        return 0.0  # Return default value
    
    # Ensure flops is a valid number
    if not isinstance(flops, (int, float)) or flops <= 0:
        print(f"Warning: Invalid FLOPs value: {flops}, using fallback")
        flops = _estimate_flops_from_input_size((1, cfg.IMG_CHANNELS, cfg.IMG_WIDTH, cfg.IMG_HEIGHT))
    
    # Calculate efficiency as FLOPs per (parameters × depth)
    try:
        efficiency = flops / (param_count * width_depth_ratio)
        return efficiency
    except (TypeError, ZeroDivisionError) as e:
        print(f"Error calculating efficiency ratio: {e}")
        print(f"flops: {flops}, param_count: {param_count}, width_depth_ratio: {width_depth_ratio}")
        return 0.0  # Return safe default value
    
def calculate_network_depth(model):
    """Calculate the depth of the network."""
    try:
        depth = 0
        
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                depth += 1
        
        return max(depth, 1)  # Ensure at least depth 1
    except Exception as e:
        print(f"Error calculating network depth: {e}")
        return 1

def get_network_width_depth_ratio(net):
    depth = 0
    width = []
    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                               nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
                               nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                               nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
                               nn.LocalResponseNorm,
                               nn.Linear, nn.Bilinear,
                               nn.Dropout, nn.Dropout2d, nn.Dropout3d,
                               nn.Embedding, nn.EmbeddingBag,
                               nn.LSTM, nn.GRU, nn.RNN,
                               nn.PReLU, nn.ReLU, nn.ReLU6, nn.RReLU,
                               nn.SELU, nn.CELU, nn.ELU, nn.GELU, nn.SiLU,
                               nn.Sigmoid, nn.Tanh, nn.LogSigmoid, nn.Softplus, nn.Softshrink,
                               nn.Softsign, nn.Tanhshrink, nn.Threshold,
                               nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                               nn.AdaptiveLogSoftmaxWithLoss, nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
                               nn.AdaptiveMaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
                               nn.FractionalMaxPool2d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                               nn.LPPool1d, nn.LPPool2d, nn.LocalResponseNorm, nn.Softmax, nn.Softmin,
                               nn.LogSoftmax, nn.Threshold)):
            depth += 1

            if hasattr(module, 'out_features'):
                width.append(module.out_features) # For Linear layers
            elif hasattr(module, 'out_channels'):
                width.append(module.out_channels) # For Conv layers

    if not width or not depth:
        return 0
    ratio = max(width)/depth
    return ratio


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_flops_robust(net, cfg: Config, device='cpu'): # TODO: Check if I could just pass Cfg instead and assign the values inside the method
    """
    Robust FLOPs calculation that handles BatchNorm issues with 1x1 spatial dimensions.
    
    Args:
        net: PyTorch network
        cfg: Configuration object
        device: Device to run calculation on (defaults to CPU for stability)
    
    Returns:
        flops: Calculated FLOPs count
    """
    
    input_shape=(1, cfg.IMG_CHANNELS, cfg.IMG_WIDTH, cfg.IMG_HEIGHT)

    # Store original training state and device
    original_training = net.training
    original_device = next(net.parameters()).device
    
    try:
        # Force network to eval mode to avoid BatchNorm training issues
        net.eval()
        
        # Ensure all BatchNorm layers are in eval mode
        for module in net.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        
        # Always use CPU for FLOPs calculation to avoid device mismatch issues
        # This is more stable and avoids CUDA memory issues
        calculation_device = 'cpu'
        
        # Move network to CPU for calculation
        net_cpu = net.to(calculation_device)
        
        # Create dummy input on CPU
        dummy_input = torch.zeros(input_shape, device=calculation_device)
        
        # Try standard FLOPs calculation
        try:
            flops = FlopCountAnalysis(net_cpu, dummy_input).total()
            
            # Validate that flops is not None and is a valid number
            if flops is None:
                raise ValueError("FLOPs calculation returned None")
            
            if not isinstance(flops, (int, float)) or flops <= 0:
                raise ValueError(f"Invalid FLOPs value: {flops}")
            
            return flops
            
        except Exception as e:
            print(f"Standard FLOPs calculation failed: {e}")
            # Fallback: estimate FLOPs based on network structure
            return _estimate_flops_fallback(net_cpu, input_shape)
        
    except Exception as e:
        print(f"FLOPs calculation failed completely: {e}")
        # Return a reasonable default based on input size
        return _estimate_flops_from_input_size(input_shape)
        
    finally: 
        # Restore original training state and device
        if original_training:
            net.train()
        else:
            net.eval()
        
        # Move network back to original device
        net.to(original_device)

def _estimate_flops_fallback(net, input_shape):
    """
    Fallback FLOPs estimation when standard calculation fails.
    Estimates based on network structure and input size.
    """
    try:
        batch_size, channels, height, width = input_shape
        total_flops = 0
        
        for name, module in net.named_modules():
            if isinstance(module, nn.Conv2d):
                # Estimate conv2d FLOPs: H * W * in_channels * out_channels * kernel_size^2
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                in_ch = module.in_channels
                out_ch = module.out_channels
                h_out = height // (module.stride[0] if isinstance(module.stride, tuple) else module.stride)
                w_out = width // (module.stride[0] if isinstance(module.stride, tuple) else module.stride)
                
                conv_flops = h_out * w_out * in_ch * out_ch * kernel_size * kernel_size
                total_flops += conv_flops
                
                # Update dimensions for next layer
                height, width = h_out, w_out
                
            elif isinstance(module, nn.Linear):
                # Estimate linear layer FLOPs
                if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                    linear_flops = module.in_features * module.out_features
                    total_flops += linear_flops
        
        return max(total_flops, 1000)  # Ensure minimum reasonable value
        
    except Exception as e:
        print(f"Fallback FLOPs estimation failed: {e}")
        return _estimate_flops_from_input_size(input_shape)

def _estimate_flops_from_input_size(input_shape):
    """
    Very basic FLOPs estimation based only on input size.
    Used as last resort when all other methods fail.
    """
    batch_size, channels, height, width = input_shape
    # Rough estimate: input_size * 1000 (typical for CNNs)
    estimated_flops = height * width * channels * 1000
    print(f"Using basic FLOPs estimate: {estimated_flops:,}")
    return estimated_flops