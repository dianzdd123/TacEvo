from llm_mutation.openrouter_mutate import openrouter_mutate
from llm_mutation.codegen_mutate import codegen_mutate
from llm_mutation.gemini_mutate import gemini_mutate
from llm_mutation.claude_mutate import claude_mutate
from core.utils import read_python_file
from prompts.resnet_prompts import get_resnet_full_prompt
from prompts.single_label_prompts import get_single_label_full_prompt
from prompts.image_regression_prompts import get_image_regression_full_prompt
# TODO: for new tasks import new preprompt functions here and implement them in the function below
from conf.config import Config
from typing import List
import os
import random
import time
import numpy as np

def generate_candidates(cfg: Config, state: dict, force_operator: str = None) -> List[dict]:

    print("##### Generating Candidates #####")
    
    logger = state.get('logger')
    generation = state.get('current_generation', -1) # assign -1 if value is not set

    # Select LLM (Mutation Method)
    if cfg.MUTATION == "openrouter":
        mutation_fn = openrouter_mutate
        print("Using OpenRouter API for mutation.")
    elif cfg.MUTATION == "codegen-350M-mono":
        mutation_fn = codegen_mutate
        print("Using local CodeGen model for mutation.")
    elif cfg.MUTATION == "gemini":
        mutation_fn = gemini_mutate
        print("Using Gemini API for mutation.")
    elif cfg.MUTATION == "claude":
        mutation_fn = claude_mutate
        print("Using Claude API for mutation.")
    else:
        error_msg = f"Unsupported mutation LLM specified in config: {cfg.MUTATION}"
        if logger:
            logger.log_error(ValueError(error_msg), "generate_candidates", generation)
        raise ValueError(error_msg)
        
    generated_nets = []
    failed_reasons = []


    ##### EVOLUTIONARY PHASE #####
    print("Phase: Evolutionary Variation")
    if not state['nets_archive']:
        error_msg = "No network archive available for evolutionary variation."
        print(error_msg)
        if logger:
            logger.log_error(Exception(error_msg), "generate_candidates", generation)
        return []
    
    # Decide operator once for the whole generation # TODO: Delete force_operator dependency after testing
    if force_operator:
        evo_operator = force_operator
    else:
        evo_operator = np.random.choice(['mutation', 'crossover'], p=[0.85, 0.15]) # 85% mutation, 15% crossover
    print(f"Performing {evo_operator} for this generation")
    
    # Initialize tracking for this generation
    if 'generation_relationships' not in state:
        state['generation_relationships'] = []
    if 'generation_operator_data' not in state:
        state['generation_operator_data'] = {
            'operator_usage': {'mutation': 0, 'crossover': 0},
            'success_rates': {'mutation': {'successful': 0, 'total': 0}, 'crossover': {'successful': 0, 'total': 0}},
            'fitness_improvements': {'mutation': [], 'crossover': []}
        }
    
    # Reset generation data
    state['generation_relationships'] = []
    state['generation_operator_data']['operator_usage'] = {'mutation': 0, 'crossover': 0}
    state['generation_operator_data']['success_rates'] = {'mutation': {'successful': 0, 'total': 0}, 'crossover': {'successful': 0, 'total': 0}}
    state['generation_operator_data']['fitness_improvements'] = {'mutation': [], 'crossover': []}

    num_nets_to_generate = cfg.NUM_NETS
    prompts_to_use = []
    temperatures_to_use = []
    operation_types = []
    base_net_code = "" # will be set within the operator block

    if evo_operator == "mutation":
        # Elitist Parent Selection
        if len(state['nets_archive']) < 3:
            n_elites = len(state['nets_archive'])
        else:
            n_elites = 3
        
        # Get top N networks based on fitness
        elite_nets = sorted(list(state['nets_archive'].values()), key=lambda x: x.fitness)[:n_elites]
        
        # Check if we have prompt archive, if not use fallback
        if state['prompt_archive']:
            elite_prompts = sorted(list(state['prompt_archive'].values()), key=lambda x: x.curiosity, reverse=True)[:n_elites]
            # Ensure we have the same number of elite prompts as networks
            actual_elites = min(len(elite_nets), len(elite_prompts))
        else:
            # Fallback: use default prompts when prompt_archive is empty
            actual_elites = len(elite_nets)
            elite_prompts = []
        
        if actual_elites == 0:
            print("Warning: No elite networks available")
            return []
        
        # Randomly select one parent from the elite group
        selection_index = random.randint(0, actual_elites - 1)
        parent_net = elite_nets[selection_index]
        
        # code for the selected parent network
        base_net_code = read_python_file(parent_net.net_path)

        # Use current temperature from experiment state for mutation operations
        current_temp = state.get('current_temperature', cfg.TEMPERATURE)
        
        # Track parent information for relationship logging
        parent_info = {
            'parent_path': parent_net.net_path,
            'parent_temperature': current_temp,
            'parent_fitness': parent_net.fitness,
            'operation_type': 'mutation'
        }

        print(f"Selected parent for mutation: {os.path.basename(parent_net.net_path)}")

        for i in range(num_nets_to_generate):
            # For mutation, use DIFFERENT prompts for each network
            # Select a RANDOM prompt from the available prompts for diversity
            if state['prompts']:
                # Randomly select a different prompt for each network
                available_prompts = list(state['prompts'])
                if len(available_prompts) > 1:
                    # Remove the previously used prompt to ensure diversity
                    used_prompts = [prompts_to_use[j] for j in range(i)]
                    available_prompts = [p for p in available_prompts if p not in used_prompts]
                    if not available_prompts:  # If all prompts used, reset
                        available_prompts = list(state['prompts'])
                
                mutation_prompt = random.choice(available_prompts)
            else:
                # Fallback to default prompt if none available
                mutation_prompt = "Refine the model architecture for better performance."
            
            # Add small temperature variation for diversity
            if i > 0:
                temp_variation = random.uniform(-0.05, 0.05)
                current_temp = max(cfg.TEMP_MIN, min(current_temp + temp_variation, cfg.TEMP_MAX))

            prompts_to_use.append(mutation_prompt)
            temperatures_to_use.append(current_temp)
            operation_types.append('mutation')
            print(f"Network {i+1}: Mutation with prompt '{mutation_prompt[:50]}' and temp {current_temp:.2f}")
    
    elif evo_operator == "crossover":
        # Generate all possible combinations of networks in the archive
        if len(state['nets_archive']) < 2:
            print("Warning: Not enough networks in archive for crossover, need at least 2")
            return []

        # Get all networks from archive and create all possible pairs
        all_nets = list(state['nets_archive'].values())
        network_combinations = []
        
        # Generate all possible pairs of networks
        for i in range(len(all_nets)):
            for j in range(i + 1, len(all_nets)):
                network_combinations.append((all_nets[i], all_nets[j]))
        
        print(f"Generated {len(network_combinations)} possible network combinations for crossover")
        
        # Handle edge cases for number of networks to generate
        if num_nets_to_generate <= len(network_combinations):
            # Use first num_nets_to_generate combinations
            selected_combinations = network_combinations[:num_nets_to_generate]
        else:
            # Use all combinations and reuse some to reach num_nets_to_generate
            selected_combinations = network_combinations.copy()
            # Reuse combinations by cycling through them
            while len(selected_combinations) < num_nets_to_generate:
                selected_combinations.extend(network_combinations[:num_nets_to_generate - len(selected_combinations)])
        
        print(f"Selected {len(selected_combinations)} combinations for this generation")
        
        # crossover prompt and a fixed temperature from a top prompt
        crossover_prompt_text = '"""Combine the above two neural networks and create a third neural network class that also inherits from nn.Module"""'
        
        # Use default temperature if prompt archive is empty
        if state['prompt_archive']:
            top_prompt = max(state['prompt_archive'].values(), key=lambda x: x.curiosity)
            crossover_temp = top_prompt.desc[1]
        else:
            crossover_temp = cfg.TEMPERATURE  # Use default temperature

        for i in range(num_nets_to_generate):
            # Get the parent pair for this network
            parent1, parent2 = selected_combinations[i]
            
            print(f"Network {i+1}: Crossover between {os.path.basename(parent1.net_path)} and {os.path.basename(parent2.net_path)} with temp {crossover_temp:.2f}")
            
            prompts_to_use.append(crossover_prompt_text)
            temperatures_to_use.append(crossover_temp)
            operation_types.append('crossover')

    else:
        raise ValueError(f"Invalid evolutionary operator: {evo_operator}")
        
    # Generate Networks with architecture-Specific Prompts
    print(f"\n Prompt Summary for Generation ")
    print(f"Evolutionary operator: {evo_operator}")
    print(f"Number of networks to generate: {len(prompts_to_use)}")
    
    # Show which prompts will be used for each network
    print("\nDetailed prompt assignment:")
    for i, (prompt, temp) in enumerate(zip(prompts_to_use, temperatures_to_use)):
        prompt_index = state['prompt_to_int'].get(prompt, "Unknown")
        prompt_preview = prompt[:60] + "" if len(prompt) > 60 else prompt
        print(f"  Network {i+1}: Prompt {prompt_index} (temp: {temp:.3f})")
        print(f"    Preview: {prompt_preview}")
    
    print(" End Prompt Summary \n")
    
    for i, (instruction_prompt, temperature) in enumerate(zip(prompts_to_use, temperatures_to_use)):
        
        # For crossover, we need to get the base code and parent info for this specific network
        if evo_operator == "crossover":
            # Get the parent pair for this network
            parent1, parent2 = selected_combinations[i]
            
            # Read codes for both parents
            parent1_code = read_python_file(parent1.net_path)
            parent2_code = read_python_file(parent2.net_path)
            
            # Base code for crossover (combo of both parents)
            current_base_net_code = f"{parent1_code}\n\n{parent2_code}"
            
            # Track parent information for relationship logging
            current_parent_info = {
                'parent_path': f"{parent1.net_path} + {parent2.net_path}",
                'parent_prompt': instruction_prompt,
                'parent_temperature': temperature,
                'parent_fitness': min(parent1.fitness, parent2.fitness),  # Use better fitness
                'operation_type': 'crossover'
            }
        else:
            # For mutation, use the base_net_code and parent_info set earlier
            current_base_net_code = base_net_code
            current_parent_info = parent_info
        
        # Architecture-specific pre-prompting for the LLM # TODO: return the pre-prompt from the prompts class instead of writing them down here
        if state['network_type'] == "resnet":
            full_prompt = get_resnet_full_prompt(instruction_prompt, current_base_net_code)
    
        elif state['network_type'] == "single_label":
            full_prompt = get_single_label_full_prompt(instruction_prompt, current_base_net_code)
            
        elif state['network_type'] == "image_regression":
            full_prompt = get_image_regression_full_prompt(instruction_prompt, current_base_net_code)
        
        # Generate the new code with adaptive temperature
        try:
            generated_code = mutation_fn(prompt=full_prompt, temperature=temperature, cfg=cfg)
                        
            # Track operator usage
            state['generation_operator_data']['operator_usage'][evo_operator] += 1
            
            generated_nets.append({
                'code': generated_code, 
                'prompt': instruction_prompt, 
                'temperature': temperature,
                'operation_type': operation_types[i],
                'parent_info': current_parent_info  # Include parent information for relationship tracking
            })
            
            prompt_index = state['prompt_to_int'].get(instruction_prompt, "Unknown")
            print(f"Generated network {i+1}/{len(prompts_to_use)} with prompt {prompt_index} (temp: {temperature:.2f})")

            # API rate limiting
            if cfg.MUTATION in ["openrouter", "gemini"]:
                print("- Pausing for 4s to respect API rate limits")
                time.sleep(4)
                
        except Exception as e:
            error_msg = f"Failed to generate network {i+1}: {e}"
            print(f"ERROR: {error_msg}")
            failed_reasons.append(error_msg)
            if logger:
                logger.log_error(e, f"network_generation_{i+1}", generation)
            continue

    print(f"Successfully generated {len(generated_nets)} candidate networks")
    
    # Log generation results
    if logger:
        logger.log_candidate_generation(generation, len(prompts_to_use), len(generated_nets), failed_reasons)
    
    return generated_nets