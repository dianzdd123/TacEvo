import os
import hashlib
import importlib.util
from core.utils import get_main_network_class, extract_and_validate_code, is_trainable
from conf.config import Config
from typing import List

# Private helper function to handle all failure cases.
def _save_failed_candidate(failure_reason: str, validation_failures: list, net_data: dict, gen_i: int, i: int, not_valid_path: str, net_path: str):
    """Handles the logic for a failed validation.
        Logging a failure and saving the problematic code to the "not_valid" directory in experiment directory."""
    
    print(f"FAIL: {failure_reason}")
    validation_failures.append(f"Candidate {i+1}: {failure_reason}")
    
    try:
        with open(not_valid_path, "w", encoding="utf-8") as f:
            f.write(f"# Failed validation - Original prompt: {net_data['prompt']}\n")
            f.write(f"# Temperature: {net_data['temperature']}\n")
            f.write(f"# Generation: {gen_i}, Candidate: {i}\n")
            f.write(f"# Reason: {failure_reason}\n\n")
            f.write(net_data['code'])
        print(f"SAVED: Failed code saved to {os.path.basename(not_valid_path)}")
    except Exception as save_error:
        print(f"ERROR: Could not save failed code: {save_error}")
    
    # deletes the temporary file from the "path_nets" directory
    if os.path.exists(net_path):
        os.remove(net_path)

def validate_candidates(cfg: Config, generated_nets: list, path_nets: str, gen_i: int, state: dict) -> List[dict]: # TODO: Added cfg as the first argument, adjust where it is called
    """
    Validates a list of generated network codes. It saves the code,
    loads the main network class, and attempts to instantiate it with
    parameters suitable for the target architecture (ResNet or sequential).
    """
    print("##### Validating Candidates #####")
    
    logger = state.get('logger')
    validation_failures = []
    valid_nets_for_training = []
    
    # Create not_valid directory in the same directory as path_nets
    not_valid_dir = os.path.join(os.path.dirname(path_nets), "not_valid")
    os.makedirs(not_valid_dir, exist_ok=True)
    print(f"Created not_valid directory: {not_valid_dir}")
    
    for i, net_data in enumerate(generated_nets):
        prompt_hash = hashlib.md5(net_data['prompt'].encode('utf-8')).hexdigest()
        net_path = os.path.normpath(f"{path_nets}/network_{gen_i}_{i}_{prompt_hash[:8]}_{net_data['temperature']:.2f}.py")
        not_valid_path = os.path.normpath(f"{not_valid_dir}/failed_{gen_i}_{i}_{prompt_hash[:8]}_{net_data['temperature']:.2f}.py")
        
        print(f"Validating candidate {i+1}/{len(generated_nets)}: {os.path.basename(net_path)}")

        # Save and validate the code
        try:
            was_saved = extract_and_validate_code(net_data['code'], file_path=net_path)
            if not was_saved:
                raise ValueError("Code could not be extracted or failed syntax validation.")
        except Exception as e:
            _save_failed_candidate(f"Code validation error - {e}", validation_failures, net_data, gen_i, i, not_valid_path, net_path)
            continue

        # Load main network class
        try:
            Net_class = get_main_network_class(net_path)
            if Net_class is None:
                raise ValueError("Could not find a valid nn.Module class in the file.")
        except Exception as e:
            _save_failed_candidate(f"Network class loading error - {e}", validation_failures, net_data, gen_i, i, not_valid_path, net_path)
            continue

        # Instantiate network with appropriate parameters
        try:
            net_instance = None
            network_type = state.get('network_type')

            if network_type == "single_label":
                net_instance = Net_class(num_classes=cfg.NUM_CLASSES)
                print(f"SUCCESS: Created SingleLabelClassCNN instance with {sum(p.numel() for p in net_instance.parameters()):,} parameters")
            
            elif network_type == "image_regression":
                net_instance = Net_class(output_size=cfg.OUTPUT_SIZE)
                print(f"SUCCESS: Created ImageRegressionCNN instance with {sum(p.numel() for p in net_instance.parameters()):,} parameters")

            elif network_type == "resnet":
                # ResNet requires finding the 'BasicBlock' helper class
                module_name = os.path.basename(net_path).replace('.py', '')
                spec = importlib.util.spec_from_file_location(module_name, net_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                block_class = getattr(module, 'BasicBlock', None)
                if not block_class:
                    raise ValueError("Could not find 'BasicBlock' class (required for ResNet).")

                net_instance = Net_class(
                    img_channels=cfg.IMG_CHANNELS,
                    num_layers=cfg.NUM_LAYERS,
                    block=block_class,
                    num_classes=cfg.NUM_CLASSES
                )
                print(f"SUCCESS: Created ResNet instance with {sum(p.numel() for p in net_instance.parameters()):,} parameters")
            
            else:
                # fallback for other unrecognized network types
                raise ValueError(f"Unsupported network_type '{network_type}' in state.")

        except Exception as e:
            _save_failed_candidate(f"Network instantiation error - {e}", validation_failures, net_data, gen_i, i, not_valid_path, net_path)
            continue

        # Validate that the network is trainable
        try:
            if not is_trainable(net_instance, cfg, state): 
                raise ValueError("Network failed the trainability check (Problem could be: forward pass or output shape).")
            
            print(f"SUCCESS: Network is trainable")
            valid_nets_for_training.append({
                "net": net_instance, 
                "path": net_path, 
                "prompt": net_data['prompt'], 
                "temp": net_data['temperature'],
                "operation_type": net_data['operation_type']
            })

        except Exception as e:
            _save_failed_candidate(f"Trainability check error - {e}", validation_failures, net_data, gen_i, i, not_valid_path, net_path)
            continue
                
    print(f"SUCCESS: Validation complete: {len(valid_nets_for_training)}/{len(generated_nets)} networks passed validation")
    print(f"Failed networks saved to: {not_valid_dir}")
    
    if logger:
        logger.log_validation_results(gen_i, len(generated_nets), len(valid_nets_for_training), validation_failures)
    
    return valid_nets_for_training