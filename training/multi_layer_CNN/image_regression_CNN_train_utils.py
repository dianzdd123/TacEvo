import torch
import torch.optim as optim
import torch.nn as nn
import time
from tqdm import tqdm
from conf.config import Config

cfg = Config()


def _get_data_for_task(task_type: str, batch_size: int):
    """Helper function to get data based on task type."""
    if task_type == "force":
        from datasets.force import get_data
    elif task_type == "pose":
        from datasets.pose.pose_loader import get_data
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    return get_data(batch_size=batch_size)

def train_net_on_gpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation loss history for analysis
    """
    
    # Get data based on current task type
    trainloader, validloader = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': []
    }

    print("------TRAINING------")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=80, leave=True)

        for data in train_pbar:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev).float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{running_loss / (train_pbar.n + 1):.4f}'})

        avg_train_loss = running_loss / len(trainloader)

        net.eval()
        valid_loss = 0.0
        valid_pbar = tqdm(validloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", ncols=80, leave=True)

        with torch.no_grad():
            for d in valid_pbar:
                data, labels = d
                data, labels = data.to(dev), labels.to(dev).float()
                target = net(data)
                loss = criterion(target.float(), labels)
                valid_loss += loss.item()
                valid_pbar.set_postfix({'Loss': f'{valid_loss / (valid_pbar.n + 1):.4f}'})

        avg_val_loss = valid_loss / len(validloader)

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)

        print(f"Epoch: {epoch+1}\t Training Loss: {avg_train_loss:.4f}")
        print(f"\t\t Valid Loss: {avg_val_loss:.4f}")

    history['final_train_loss'] = history['train_losses'][-1]
    history['final_val_loss'] = history['val_losses'][-1]

    return history


def train_net_on_cpu(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM):
    """
    Returns a dictionary containing the training and validation loss history for analysis
    """
    
    # Get data based on current task type
    trainloader, validloader = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cpu")
    print(f"The network will train on {dev} device")

    net.to(dev)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    history = {
        'epochs': [],
        'train_losses': [],
        'val_losses': []
    }

    print("------TRAINING------")
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        train_pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs} [Train]", ncols=80, leave=True)

        for data in train_pbar:
            inputs, labels = data
            inputs, labels = inputs.to(dev), labels.to(dev).float()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{running_loss / (train_pbar.n + 1):.4f}'})

        avg_train_loss = running_loss / len(trainloader)

        net.eval()
        valid_loss = 0.0
        valid_pbar = tqdm(validloader, desc=f"Epoch {epoch+1}/{epochs} [Valid]", ncols=80, leave=True)
        with torch.no_grad():
            for d in valid_pbar:
                data, labels = d
                data, labels = data.to(dev), labels.to(dev).float()
                target = net(data)
                loss = criterion(target.float(), labels)
                valid_loss += loss.item()
                valid_pbar.set_postfix({'Loss': f'{valid_loss / (valid_pbar.n + 1):.4f}'})

        avg_val_loss = valid_loss / len(validloader)

        history['epochs'].append(epoch + 1)
        history['train_losses'].append(avg_train_loss)
        history['val_losses'].append(avg_val_loss)

        print(f"Epoch: {epoch+1}\t Training Loss: {avg_train_loss:.4f}")
        print(f"\t\t Valid Loss: {avg_val_loss:.4f}")

    history['final_train_loss'] = history['train_losses'][-1]
    history['final_val_loss'] = history['val_losses'][-1]

    return history


def forward_pass_on_cpu(net, batch_size=cfg.BATCH_SIZE):
    """
    Performs a forward pass on the CPU for regression.
    """
    
    # Get data based on current task type
    trainloader, _ = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    criterion = nn.MSELoss() # Using MSELoss
    running_loss = 0.0
    start = time.time()
    net.eval()
    forward_pbar = tqdm(trainloader, desc="Forward Pass [CPU]", ncols=100)

    with torch.no_grad():
        for data, target in forward_pbar:
            data, target = data.to(dev), target.to(dev).float()
            output = net(data)
            loss = criterion(output.float(), target)
            running_loss += loss.item()
            current_loss = running_loss / (forward_pbar.n + 1)
            forward_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

    inference_time = time.time() - start
    avg_loss = running_loss / len(trainloader)
    print(f"\tForward Pass -> Loss: {avg_loss:.4f}\t Inference Time: {inference_time:.2f}s")
    return avg_loss, inference_time


def forward_pass_on_gpu(net, batch_size=cfg.BATCH_SIZE):
    """
    Performs a forward pass on the GPU for regression.
    """
    
    # Get data based on current task type
    trainloader, _ = _get_data_for_task(cfg.TASK_TYPE, batch_size)
    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"The network will perform a forward pass on {dev} device")

    net.to(dev)
    criterion = nn.MSELoss() # Using MSELoss
    running_loss = 0.0
    start = time.time()
    net.eval()
    forward_pbar = tqdm(trainloader, desc="Forward Pass [GPU]", ncols=80, leave=True)

    with torch.no_grad():
        for data, target in forward_pbar:
            data, target = data.to(dev), target.to(dev).float()
            output = net(data)
            loss = criterion(output.float(), target)
            running_loss += loss.item()
            current_loss = running_loss / (forward_pbar.n + 1)
            forward_pbar.set_postfix({'Loss': f'{current_loss:.4f}'})

    inference_time = time.time() - start
    avg_loss = running_loss / len(trainloader)
    print(f"\tForward Pass -> Loss: {avg_loss:.4f}\t Inference Time: {inference_time:.2f}s")
    return avg_loss, inference_time