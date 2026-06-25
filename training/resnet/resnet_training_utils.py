import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm

from datasets.CIFAR_10 import get_data
from conf.config import Config

cfg = Config()

def train_net_on_gpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation history for analysis
    """
    trainloader, validloader = get_data(batch_size=batch_size)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum) 

    history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    print("------TRAINING------")
    for epoch in range(epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=80, leave=True)

        for data in train_pbar:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'Loss': f'{running_loss / (train_pbar.n + 1):.4f}', 'Acc': f'{100 * correct / total:.2f}%'})

        avg_train_loss = running_loss / len(trainloader)
        training_accuracy = 100 * correct / total

        net.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        valid_pbar = tqdm(validloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", ncols=80, leave=True)

        with torch.no_grad():
            for d in valid_pbar:
                data, labels = d
                data, labels = data.to(dev), labels.to(dev)
                target = net(data)
                loss = criterion(target, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(target.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                valid_pbar.set_postfix({'Loss': f'{valid_loss / (valid_pbar.n + 1):.4f}', 'Acc': f'{100 * valid_correct / valid_total:.2f}%'})

        avg_val_loss = valid_loss / len(validloader)
        valid_accuracy = 100 * valid_correct / valid_total

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['train_accs'].append(training_accuracy)
        history['val_losses'].append(avg_val_loss)
        history['val_accs'].append(valid_accuracy)

        print(f"Epoch: {epoch+1}\t Training Loss: {avg_train_loss:.4f}\t Train Acc: {training_accuracy:.2f}%")
        print(f"\t\t Valid Loss: {avg_val_loss:.4f}\t\t Valid Acc: {valid_accuracy:.2f}%")

    history['final_train_loss'] = history['train_losses'][-1]
    history['final_train_acc'] = history['train_accs'][-1]
    history['final_val_loss'] = history['val_losses'][-1]
    history['final_val_acc'] = history['val_accs'][-1]

    return history


def train_net_on_cpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation history for analysis
    """
    trainloader, validloader = get_data(batch_size=batch_size)
    dev = torch.device("cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    print("------TRAINING------")
    for epoch in range(epochs):
        net.train()
        running_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=80, leave=True)

        for data in train_pbar:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            train_pbar.set_postfix({'Loss': f'{running_loss / (train_pbar.n + 1):.4f}', 'Acc': f'{100 * correct / total:.2f}%'})

        avg_train_loss = running_loss / len(trainloader)
        training_accuracy = 100 * correct / total

        net.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        valid_pbar = tqdm(validloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", ncols=80, leave=True)

        with torch.no_grad():
            for d in valid_pbar:
                data, labels = d
                data, labels = data.to(dev), labels.to(dev)
                target = net(data)
                loss = criterion(target, labels)
                valid_loss += loss.item()
                _, predicted = torch.max(target.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                valid_pbar.set_postfix({'Loss': f'{valid_loss / (valid_pbar.n + 1):.4f}', 'Acc': f'{100 * valid_correct / valid_total:.2f}%'})

        avg_val_loss = valid_loss / len(validloader)
        valid_accuracy = 100 * valid_correct / valid_total

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['train_accs'].append(training_accuracy)
        history['val_losses'].append(avg_val_loss)
        history['val_accs'].append(valid_accuracy)

        print(f"Epoch: {epoch+1}\t Training Loss: {avg_train_loss:.4f}\t Train Acc: {training_accuracy:.2f}%")
        print(f"\t\t Valid Loss: {avg_val_loss:.4f}\t\t Valid Acc: {valid_accuracy:.2f}%")

    history['final_train_loss'] = history['train_losses'][-1]
    history['final_train_acc'] = history['train_accs'][-1]
    history['final_val_loss'] = history['val_losses'][-1]
    history['final_val_acc'] = history['val_accs'][-1]

    return history


def forward_pass_on_cpu(net, batch_size=cfg.BATCH_SIZE): 
    """
    Performs a forward pass on the CPU to get loss and inference speed.
    """
    trainloader, _ = get_data(batch_size=batch_size)
    dev = torch.device("cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    start = time.time()
    net.eval()
    
    # Forward pass progress bar
    forward_pbar = tqdm(trainloader, desc="Forward Pass [CPU]", ncols=100)
    
    with torch.no_grad():
        for data, target in forward_pbar:
            data, target = data.to(dev), target.to(dev)
            output = net(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (forward_pbar.n + 1)
            current_acc = 100 * correct / total
            forward_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    inference_time = time.time() - start
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(trainloader)

    print(
        f"\tForward Pass -> Loss: {avg_loss:.4f}\t Accuracy: {accuracy:.2f}%\t Inference Time: {inference_time:.2f}s"
    )

    return avg_loss, inference_time


def forward_pass_on_gpu(net, batch_size=cfg.BATCH_SIZE): 
    """
    Performs a forward pass on the GPU to get loss and inference speed.
    """
    trainloader, _ = get_data(batch_size=batch_size)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0

    start = time.time()
    net.eval()
    
    # Forward pass progress bar
    forward_pbar = tqdm(trainloader, desc="Forward Pass [GPU]", ncols=80, leave=True)
    
    with torch.no_grad():
        for data, target in forward_pbar:
            data, target = data.to(dev), target.to(dev)
            output = net(data)
            loss = criterion(output, target)
            running_loss += loss.item()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (forward_pbar.n + 1)
            current_acc = 100 * correct / total
            forward_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    inference_time = time.time() - start
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(trainloader)

    print(
        f"\tForward Pass -> Loss: {avg_loss:.4f}\t Accuracy: {accuracy:.2f}%\t Inference Time: {inference_time:.2f}s"
    )

    return avg_loss, inference_time


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
    Detect all unique layer types in a neural network model.
    """
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
        
        print(f"Detected {len(unique_layers)} unique layer types: {unique_layers}")
        return unique_layers
        
    except Exception as e:
        print(f"Error in detect_layers: {e}")
        return []