import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm
#from datasets import get_data # TODO: Change this once different datasets are implemented
from conf.config import Config

cfg = Config()



def _get_data_for_task(task_type: str, batch_size: int):
    """Helper function to get data based on task type."""
    if task_type == "shape":
        from datasets.shape import get_data
    elif task_type == "CIFAR-10":
        from datasets.CIFAR_10 import get_data
    elif task_type == "grating":
        from datasets.grating import get_data
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return get_data(batch_size=batch_size)

def train_net_on_gpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation history for analysis
    """
    
    # Get data based on current task type
    trainloader, validloader = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    
    # Use CrossEntropyLoss instead of NLLLoss for better numerical stability
    # CrossEntropyLoss combines LogSoftmax + NLLLoss internally
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    # History tracking
    history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    # # Validate network architecture before training
    # _validate_network_architecture(net, trainloader, dev)
    
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
            
            
            # CrossEntropyLoss expects raw logits, no need for LogSoftmax
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
                                
                # CrossEntropyLoss expects raw logits
                loss = criterion(target, labels)
                valid_loss += loss.item()
                
                _, predicted = torch.max(target.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
                
                valid_pbar.set_postfix({'Loss': f'{valid_loss / (valid_pbar.n + 1):.4f}', 'Acc': f'{100 * valid_correct / valid_total:.2f}%'})

        avg_val_loss = valid_loss / len(validloader)
        validation_accuracy = 100 * valid_correct / valid_total

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['train_accs'].append(training_accuracy)
        history['val_losses'].append(avg_val_loss)
        history['val_accs'].append(validation_accuracy)

        print(f"Epoch: {epoch+1}\t Training Loss: {avg_train_loss:.4f}\t Training Acc: {training_accuracy:.2f}%")
        print(f"\t\t Valid Loss: {avg_val_loss:.4f}\t Valid Acc: {validation_accuracy:.2f}%")

    history['final_train_loss'] = history['train_losses'][-1]
    history['final_val_loss'] = history['val_losses'][-1]
    history['final_train_acc'] = history['train_accs'][-1]
    history['final_val_acc'] = history['val_accs'][-1]

    return history


def train_net_on_cpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation history for analysis
    """
    
    # Get data based on current task type
    trainloader, validloader = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    # Use CrossEntropyLoss instead of NLLLoss for better numerical stability
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    history = {
        'epochs': [],
        'train_losses': [],
        'train_accs': [],
        'val_losses': [],
        'val_accs': []
    }

    # # Validate network architecture before training
    # _validate_network_architecture(net, trainloader, dev)

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

    trainloader, _ = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    dev = torch.device("cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    # Use CrossEntropyLoss instead of NLLLoss for better numerical stability
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    net.eval()

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

            current_loss = running_loss / (forward_pbar.n + 1)
            current_acc = 100 * correct / total
            forward_pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

    inference_time = time.time() - start
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(trainloader)

    print(f"\tForward Pass -> Loss: {avg_loss:.4f}\t Accuracy: {accuracy:.2f}%\t Inference Time: {inference_time:.2f}s")
    return avg_loss, inference_time


def forward_pass_on_gpu(net, batch_size=cfg.BATCH_SIZE):
    """
    Performs a forward pass on the GPU to get loss and inference speed.
    """

    trainloader, _ = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    # Use CrossEntropyLoss instead of NLLLoss for better numerical stability
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    net.eval()

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

            current_loss = running_loss / (forward_pbar.n + 1)
            current_acc = 100 * correct / total
            forward_pbar.set_postfix({'Loss': f'{current_loss:.4f}', 'Acc': f'{current_acc:.2f}%'})

    inference_time = time.time() - start
    accuracy = 100 * correct / total
    avg_loss = running_loss / len(trainloader)

    print(f"\tForward Pass -> Loss: {avg_loss:.4f}\t Accuracy: {accuracy:.2f}%\t Inference Time: {inference_time:.2f}s")
    return avg_loss, inference_time
