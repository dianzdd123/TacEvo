import torch.nn as nn
import numpy as np
from conf.config import Config

# Architectural Diversity Descriptor for QD Optimization

def architectural_diversity_score(model, cfg: Config):
    """
    Calculate normalized architectural diversity score.
    Range: [0, 1] - continuous
    0 = very uniform architecture, 1 = very diverse architecture
    """
    try:
        operation_entropy = calculate_operation_type_entropy(model)
        connection_density = calculate_connection_density(model)
        layer_heterogeneity = calculate_layer_variance(model)
        
        diversity_score = (operation_entropy + connection_density + layer_heterogeneity) / 3
        return np.clip(diversity_score, 0, 1)
    except Exception as e:
        print(f"Error calculating architectural diversity: {e}")
        return cfg.DEFAULT_DESCRIPTOR_VALUE  # Default middle value

def calculate_operation_type_entropy(model):
    """Calculate entropy of different operation types in the network."""
    try:
        operation_counts = {}
        
        for module in model.modules():
            op_type = type(module).__name__
            # Skip certain generic containers
            if op_type not in ['Sequential', 'ModuleList', 'ModuleDict']:
                operation_counts[op_type] = operation_counts.get(op_type, 0) + 1
        
        # Remove the top-level module itself and other non-operation modules
        exclude_types = [type(model).__name__, 'ReLU', 'Dropout', 'Identity']
        for exclude_type in exclude_types:
            if exclude_type in operation_counts and operation_counts[exclude_type] >= 1:
                # Keep at least one instance to avoid empty dict
                if len(operation_counts) > 1:
                    operation_counts[exclude_type] = max(1, operation_counts[exclude_type] - 1)
                    if operation_counts[exclude_type] == 0:
                        del operation_counts[exclude_type]
        
        # Calculate entropy
        total_ops = sum(operation_counts.values())
        if total_ops <= 1 or len(operation_counts) <= 1:
            return 0.0
        
        entropy = 0.0
        for count in operation_counts.values():
            if count > 0:
                prob = count / total_ops
                entropy -= prob * np.log2(prob)
        
        # Normalize entropy by maximum possible entropy (log2 of unique operations)
        max_entropy = np.log2(len(operation_counts)) if len(operation_counts) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return np.clip(normalized_entropy, 0, 1)
    except Exception as e:
        print(f"Error calculating operation entropy: {e}")
        return 0.0

def calculate_connection_density(model):
    """Calculate connection density of the network."""
    try:
        total_connections = 0
        total_possible_connections = 0
        layers = []
        
        # Count layers and their output sizes
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                layers.append(module)
        
        if len(layers) <= 1:
            return 0.0
        
        # For simplicity, assume sequential connections plus possible skip connections
        # Sequential connections
        sequential_connections = len(layers) - 1
        
        # Possible skip connections (any layer to any later layer)
        total_possible_connections = len(layers) * (len(layers) - 1) // 2
        
        # Count actual skip connections by looking for residual/skip patterns
        skip_connections = 0
        for module in model.modules():
            # Look for modules that might indicate skip connections
            if hasattr(module, 'shortcut') or hasattr(module, 'downsample'):
                skip_connections += 1
            # Check for addition operations that might indicate residual connections
            elif 'residual' in type(module).__name__.lower():
                skip_connections += 1
        
        total_connections = sequential_connections + skip_connections
        
        if total_possible_connections == 0:
            return 0.0
        
        density = total_connections / total_possible_connections
        return np.clip(density, 0, 1)
    except Exception as e:
        print(f"Error calculating connection density: {e}")
        return 0.0

def calculate_layer_variance(model):
    """Calculate variance in layer properties."""
    try:
        layer_properties = []
        
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                # For conv layers, consider kernel size, channels, padding, stride, groups
                try:
                    properties = [
                        module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                        module.in_channels,
                        module.out_channels,
                        module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                        module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                        getattr(module, 'groups', 1)  # Default to 1 if groups not present
                    ]
                    layer_properties.append(properties)
                except Exception as e:
                    # If any property extraction fails, use defaults
                    layer_properties.append([3, 256, 256, 1, 1, 1])  # Default conv properties
                    
            elif isinstance(module, nn.Linear):
                # For linear layers, consider input and output features
                try:
                    properties = [1, module.in_features, module.out_features, 0, 0, 1]  # Pad to same length
                    layer_properties.append(properties)
                except Exception:
                    layer_properties.append([1, 512, 10, 0, 0, 1])  # Default linear properties
        
        if len(layer_properties) <= 1:
            return 0.0
        
        # Calculate variance for each property type
        try:
            layer_properties = np.array(layer_properties, dtype=np.float32)
            property_variances = []
            
            for i in range(layer_properties.shape[1]):
                prop_values = layer_properties[:, i]
                unique_values = np.unique(prop_values)
                
                if len(unique_values) > 1:  # Only calculate variance if there's variation
                    variance = np.var(prop_values)
                    mean_val = np.mean(prop_values)
                    
                    if mean_val > 0 and variance > 0:
                        # Coefficient of variation
                        cv = np.sqrt(variance) / mean_val
                        property_variances.append(cv)
            
            if not property_variances:
                return 0.0
            
            # Average coefficient of variation across all properties
            avg_cv = np.mean(property_variances)
            
            # Normalize to [0, 1] range 
            normalized_variance = np.clip(avg_cv, 0, 1)
            
            return normalized_variance
            
        except Exception as e:
            print(f"Error in variance calculation: {e}")
            return 0.0
            
    except Exception as e:
        print(f"Error calculating layer variance: {e}")
        return 0.0