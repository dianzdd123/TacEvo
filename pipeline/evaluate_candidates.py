import os 
from conf.config import Config
import torch.nn as nn
from core.utils import detect_layers, transfer_weights
# FIXME: This should work since I have placed them in the __init__.py package file
# from training import single_label_CNN_train_utils, image_regression_CNN_train_utils, resnet_training_utils


def evaluate_candidates(valid_nets: list, cfg: Config, curios_net: nn.Module, state: dict):
    print("##### Evaluating Candidates #####")
    fitness_results = []

    # Use architecture-specific training functions
    if cfg.NETWORK_TYPE == "resnet":
        from training.resnet.resnet_training_utils import train_net_on_gpu, train_net_on_cpu, forward_pass_on_gpu, forward_pass_on_cpu
        eval_fn_forward = forward_pass_on_gpu if cfg.DEVICE == "cuda" else forward_pass_on_cpu
        eval_fn_train = train_net_on_gpu if cfg.DEVICE == "cuda" else train_net_on_cpu
    elif cfg.NETWORK_TYPE == "single_label":
        from training.multi_layer_CNN.single_label_CNN_train_utils import train_net_on_gpu, train_net_on_cpu, forward_pass_on_gpu, forward_pass_on_cpu
        eval_fn_forward = forward_pass_on_gpu if cfg.DEVICE == "cuda" else forward_pass_on_cpu
        eval_fn_train = train_net_on_gpu if cfg.DEVICE == "cuda" else train_net_on_cpu
    elif cfg.NETWORK_TYPE == "image_regression":
        from training.multi_layer_CNN.image_regression_CNN_train_utils import train_net_on_gpu, train_net_on_cpu, forward_pass_on_gpu, forward_pass_on_cpu
        eval_fn_forward = forward_pass_on_gpu if cfg.DEVICE == "cuda" else forward_pass_on_cpu
        eval_fn_train = train_net_on_gpu if cfg.DEVICE == "cuda" else train_net_on_cpu
    else:
        raise ValueError(f"Unsupported network type: {cfg.NETWORK_TYPE}")

    for i, net_data in enumerate(valid_nets):
        net = net_data["net"]
        print(f"Evaluating network {i+1}/{len(valid_nets)}: {os.path.basename(net_data['path'])}")
        
        try:
            loss = float('inf')
            accuracy = 0.0
            infer_time = 0.0
            training_results = {}

            # Main evaluation phase
            print("Running in main evaluation mode (full training)")
            # Transfer weights from parent network
            if curios_net is not None:
                print("Transferring weights from parent network")
                try:
                    for layer in detect_layers(curios_net):
                        try: 
                            net = transfer_weights(curios_net, net, layer)
                        except Exception as e:
                            pass # Silently ignore layers that can't be transferred
                except Exception as e:
                    print(f"Warning: Weight transfer failed: {e}")
            
            # Run forward pass for inference speed & low-fidelity training
            _, infer_time = eval_fn_forward(net, cfg.BATCH_SIZE)
            training_results = eval_fn_train(net, epochs=cfg.NET_TRAINING_EPOCHS, lr=cfg.LEARNING_RATE, batch_size=cfg.BATCH_SIZE, momentum=cfg.MOMENTUM)
            
            loss = training_results['final_val_loss']
            if state['network_type'] != "image_regression": # accuracy isnt used for regression
                accuracy = training_results['final_val_acc']
            
            print(f"Evaluation complete -> Loss: {loss:.4f}, Speed: {infer_time:.2f}s")
            
            # Save network fitness results
            result = {
                "loss": loss, 
                "accuracy": accuracy,
                "inference_time": infer_time,
                "training_results": training_results,
                "valid": True,
                "added_to_archive": False, # Will be set later
                **net_data
            }
            
            # Track parent-child relationship and operator success
            if 'parent_info' in net_data:
                parent_info = net_data['parent_info']
                operation_type = parent_info['operation_type']
                
                # Update operator success tracking
                state['generation_operator_data']['success_rates'][operation_type]['total'] += 1
                
                # Track fitness improvement
                fitness_improvement = parent_info['parent_fitness'] - loss
                state['generation_operator_data']['fitness_improvements'][operation_type].append(fitness_improvement)
                
                # Add to parent-child relationships
                relationship = {
                    'child_path': net_data['path'],
                    'child_fitness': loss,
                    'parent_path': parent_info['parent_path'],
                    'parent_prompt': parent_info['parent_prompt'],
                    'parent_temperature': parent_info['parent_temperature'],
                    'parent_fitness': parent_info['parent_fitness'],
                    'operation_type': operation_type,
                    'success': True  # Will be updated later if added to archive
                }
                state['generation_relationships'].append(relationship)
            # For image regression, the accuracy key is not needed
            if state['network_type'] == "image_regression":
                del result['accuracy']

            fitness_results.append(result)
            
        except Exception as e:
            print(f"ERROR: Evaluation failed for {os.path.basename(net_data['path'])}")
            print(f"Reason: {e}")
    
    print(f"SUCCESS: Evaluation complete: {len(fitness_results)} networks evaluated")
    return fitness_results