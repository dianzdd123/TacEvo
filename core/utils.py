import matplotlib.pyplot as plt
import ast
import importlib.util
import inspect
import os
import csv
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from typing import Type
from fvcore.nn import FlopCountAnalysis
import autopep8 # Import the new library
from conf.config import Config

cfg = Config()

class Net(): 
    """If the system fails to load a proper network class, 
    it can instantiate this dummy class instead of crashing"""
    pass

def get_max_curiosity(species_list, n):
    species_list_sorted = sorted(species_list, key=lambda x: x.curiosity, reverse=True)
    return species_list_sorted[:n]

def get_max_fitness(species_list, n):
    species_list_sorted = sorted(species_list, key=lambda x: x.fitness, reverse=False)
    return species_list_sorted[:n]

def read_python_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_and_validate_code(code_string: str, file_path: str): # TODO: Fix where this is called to stop passing "prompt" as an argument (prompt used to be an arg and got removed)
    """
    Extracts Python class definitions from generated code and saves to file.

    Returns True on success, False otherwise.
    """
    
    # Extract code from markdown blocks if present
    raw_code = ""
    if "```python" in code_string:
        try:
            # Extract code from markdown blocks
            code_block = code_string.split("```python")[1]
            raw_code = code_block.split("```")[0]
        except IndexError:
            print(" - FAIL: Could not extract code from markdown block.")
            return False
    elif "```" in code_string:
        try:
            # Handle bare markdown fences without language specification
            parts = code_string.split("```")
            if len(parts) >= 3:
                # Take the middle part (the code between the fences)
                raw_code = parts[1]
            else:
                # Handle case where there's only a trailing ```
                raw_code = parts[0]
        except IndexError:
            print(" - FAIL: Could not extract code from markdown block.")
            return False
    else:
        # If no markdown, use the entire string
        raw_code = code_string
            
    raw_code = raw_code.strip()
    if not raw_code:
        print(" - FAIL: No code found to extract.")
        return False

    # Clean up the code - remove excessive blank lines
    lines = raw_code.split('\n')
    cleaned_lines = []
    prev_blank = False
    
    for line in lines:
        if line.strip() == "":
            if not prev_blank:
                cleaned_lines.append(line)
                prev_blank = True
        else:
            cleaned_lines.append(line)
            prev_blank = False
    
    final_code = '\n'.join(cleaned_lines)

    # File Write Step
    try:
        # Write the code to the specified file with imports
        full_code_for_file = (
            "import torch\n"
            "import torch.nn as nn\n"
            "import torch.nn.functional as F\n"
            "from torch import Tensor\n"
            "from typing import Type\n\n"
            f"{final_code}"
        )
        
        with open(os.path.normpath(file_path), "w", encoding="utf-8") as f:
            f.write(full_code_for_file)
        print(f" - SUCCESS (File Write): Code written to {file_path}")
        return True
    except Exception as e:
        print(f" - ERROR (File Write): Failed to write to {file_path}. Reason: {e}")
        return False


def get_main_network_class(filename):
    """
    Loads a Python module from a file path and returns the main class
    that inherits from nn.Module.

    assumes the main network class is the last one defined in the file. # NOTE: ResNet class (for Resnet) SimpleCNN for multi_layer_CNN
    """
    # A list to hold all found nn.Module classes
    found_classes = []
    
    try:
        # Create a unique module name from the file path
        module_name = os.path.basename(filename).replace('.py', '')
        
        # Load the module from the file
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None:
            print(f" ERROR (Import): Could not create module spec from {filename}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Inspect the module to find all nn.Module subclasses
        for name, obj in inspect.getmembers(module):
            # Ensure it's a class, a subclass of nn.Module, and not nn.Module itself
            if inspect.isclass(obj) and issubclass(obj, nn.Module) and obj is not nn.Module:
                found_classes.append(obj)
                print(f" FOUND: Class '{name}' in module")
                
    except Exception as e:
        print(f" ERROR (Import): Failed to load or inspect module {os.path.basename(filename)}. Reason: {e}")
        return None
    
    # If classes were found, return the main network class
    if found_classes:
        # Look for the main network class by name patterns
        main_class = None
        for cls in found_classes:
            class_name = cls.__name__
            # Prioritize classes that are likely the main network
            if any(pattern in class_name.lower() for pattern in ['cnn', 'resnet', 'regression', 'classification']):
                main_class = cls
                break
        
        # If no specific pattern found, use the last class
        if main_class is None:
            main_class = found_classes[-1]
            
        print(f" SUCCESS (Import): Found main network class '{main_class.__name__}'")
        return main_class  
    
    # Return None if no suitable class was found
    print(f" FAIL (Import): No nn.Module subclass found in {os.path.basename(filename)}")
    return None


def csv_writer(results, output_file):
    with open(output_file, 'a', newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(results)

# def is_trainable(net, input_shape=(1, cfg.IMG_CHANNELS, cfg.IMG_WIDTH, cfg.IMG_HEIGHT), num_classes=cfg.NUM_CLASSES): # TODO: this has the num_classes as an arg, this could cause problems with the regression task
#     zeros = torch.zeros(input_shape)  # input tensor of shape (batch_size, channels, height, width)

#     # Check that we can pass a dummy input through the network without errors.
#     try:
#         output = net(zeros)
#     except Exception as e:
#         return False
    
#     # Network output shape must match number of classes in CIFAR-10.
#     if output.shape != (1, num_classes):
#         return False

#     return True

# import torch
# from conf.config import Config # Assuming cfg is an instance of Config

def is_trainable(net: torch.nn.Module, cfg: Config, state: dict) -> bool:
    """
    Checks if a network is trainable by performing a forward pass with a dummy tensor
    and validating the output shape based on the network type.
    """

    # Create a dummy input tensor based on config
    input_shape = (1, cfg.IMG_CHANNELS, cfg.IMG_WIDTH, cfg.IMG_HEIGHT)
    dummy_input = torch.zeros(input_shape)

    # Check forward pass
    try:
        # Set network to eval mode to avoid BatchNorm issues with single sample
        net.eval()
        with torch.no_grad():
            output = net(dummy_input)
    except Exception as e:
        print(f"Trainability FAIL: Forward pass failed with error: {e}")
        return False

    # Check output shape is correct for the network type
    network_type = state.get('network_type')
    if network_type in ["single_label", "resnet"]:
        expected_shape = (1, cfg.NUM_CLASSES)
        if output.shape != expected_shape:
            print(f"Trainability FAIL: Expected output shape {expected_shape}, but got {output.shape}")
            return False
    elif network_type == "image_regression":
        expected_shape = (1, cfg.OUTPUT_SIZE)
        if output.shape != expected_shape:
            print(f"Trainability FAIL: Expected output shape {expected_shape}, but got {output.shape}")
            return False
    else:
        # If the network type is unknown -> cannot validate the shape
        print(f"Trainability WARNING: Unknown network_type '{network_type}'. Skipping output shape validation.")
        pass

    return True


def transfer_weights(netA, netB, layer_type): # TODO: test this between pregen models
    """
    Transfer weights from layers of a specific type in netA to corresponding layers in netB.
    Generalized to handle different weight tensor dimensions (Different architecture comptability).
    """
    try:
        # Get the parameters of the two networks
        paramsA = netA.state_dict()
        paramsB = netB.state_dict()
        
        transferred_count = 0
        
        # Transfer the weights from the layers of the given type in netA to the corresponding layers in netB
        for name, module in netA.named_modules():
            if type(module).__name__ == layer_type:
                # Check if the corresponding layer exists in netB
                if name + '.weight' in paramsB and name + '.bias' in paramsB:
                    try:
                        # Handle different weight tensor dimensions
                        if len(paramsB[name + '.weight'].shape) == 5:
                            paramsB[name + '.weight'][:, :, :, :, :] = paramsA[name + '.weight'][:, :, :, :, :]
                        elif len(paramsB[name + '.weight'].shape) == 4:
                            paramsB[name + '.weight'][:, :, :, :] = paramsA[name + '.weight'][:, :, :, :]
                        elif len(paramsB[name + '.weight'].shape) == 3:
                            paramsB[name + '.weight'][:, :, :] = paramsA[name + '.weight'][:, :, :]
                        elif len(paramsB[name + '.weight'].shape) == 2:
                            paramsB[name + '.weight'][:, :] = paramsA[name + '.weight'][:, :]
                        elif len(paramsB[name + '.weight'].shape) == 1:
                            paramsB[name + '.weight'][:] = paramsA[name + '.weight'][:]
                        
                        # Transfer bias if it exists
                        if name + '.bias' in paramsA:
                            paramsB[name + '.bias'][:] = paramsA[name + '.bias'][:]
                        
                        transferred_count += 1
                        
                    except Exception as e:
                        pass  # Could not transfer weights
                        continue
                else:
                    pass  # Layer not found in target network
        
        # Set the state dict of netB to the updated parameters
        netB.load_state_dict(paramsB)
        
        if transferred_count > 0:
            pass  # Transferred weights successfully
        else:
            pass  # No layers were transferred
            
        return netB
        
    except Exception as e:
        print(f"Error in transfer_weights: {e}")
        return netB


def detect_layers(model):
    """
    Detect all unique layer types in a neural network model.    """
    layers = []

    def detect_layers_recursively(module):
        for child in module.children():
            if isinstance(child, nn.Sequential):
                detect_layers_recursively(child)
            elif isinstance(child, nn.Module):
                # Get the actual class name, not the module name
                layer_name = child.__class__.__name__
                layers.append(layer_name)
                # Recursively check children of this module (important for ResNet blocks)
                detect_layers_recursively(child)

    try:
        detect_layers_recursively(model)
        unique_layers = list(set(layers))
        
        # Sort for consistent output
        unique_layers.sort()
        
        #print(f"Detected {len(unique_layers)} unique layer types: {unique_layers}")
        return unique_layers
        
    except Exception as e:
        print(f"Error in detect_layers: {e}")
        return []