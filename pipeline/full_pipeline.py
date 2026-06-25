from conf.config import Config
from datetime import datetime
from pipeline.initialize_expriment import initialize_experiment
from pipeline.archive_update import update_archives
from pipeline.validate_candidates import validate_candidates
from pipeline.evaluate_candidates import evaluate_candidates
from pipeline.generate_candidates import generate_candidates
from pipeline.logging_utils import save_experiment_summary, update_experiment_summary
from pipeline.archive_utils import save_archive_results
from plotter.live_plotter import LivePlotter, create_plots
from pregens.init_archive_pregen import init_archive_pregen
import random


def generate_new_random_prompts(experiment_state, cfg):
   
    print("Generating new random prompts for this generation")
    
    # Get the base prompt class based on network type
    if cfg.NETWORK_TYPE == "resnet":
        from prompts.resnet_prompts import ResNetPrompts
        prompt_class = ResNetPrompts()
    elif cfg.NETWORK_TYPE == "image_regression":
        from prompts.image_regression_prompts import ImageRegressionPrompts
        prompt_class = ImageRegressionPrompts()
    elif cfg.NETWORK_TYPE == "single_label":
        from prompts.single_label_prompts import SingleLabelPrompts
        prompt_class = SingleLabelPrompts()
    else:
        raise ValueError(f"Invalid network type: {cfg.NETWORK_TYPE}")
    
    # Get all available prompts
    all_prompts = prompt_class.get_prompts()
    
    # Randomly select NUM_NETS prompts for this generation
    num_prompts_needed = cfg.NUM_NETS
    
    if len(all_prompts) < num_prompts_needed:
        print(f"Warning: Only {len(all_prompts)} prompts available, but {num_prompts_needed} needed. Some prompts will be repeated.")
        # If we don't have enough unique prompts, allow repetition
        selected_prompts = random.choices(all_prompts, k=num_prompts_needed)
    else:
        # Randomly sample without replacement
        selected_prompts = random.sample(all_prompts, num_prompts_needed)
    
    # Update the experiment state with new prompts
    experiment_state['prompts'] = selected_prompts
    experiment_state['probabilities'] = [1.0 / len(selected_prompts)] * len(selected_prompts)
    experiment_state['prompt_to_int'] = {p: i for i, p in enumerate(selected_prompts)}
    experiment_state['int_to_prompt'] = {i: p for i, p in enumerate(selected_prompts)}
    
    print(f"Generated {len(selected_prompts)} new random prompts for generation")
    print(f"Prompt indices: {list(range(len(selected_prompts)))}")
    
    # Show a preview of the prompts (first 100 characters)
    print("\n Prompt Preview ")
    for i, prompt in enumerate(selected_prompts):
        preview = prompt[:100] + "" if len(prompt) > 100 else prompt
        print(f"Prompt {i}: {preview}")
    print(" End Prompt Preview \n")
    
    return selected_prompts


def log_final_prompt_diversity_summary(experiment_state, cfg):

    print("\n" + "="*50)
    print("FINAL PROMPT DIVERSITY SUMMARY")
    print("="*50)
    
    # Count total unique prompts used across all generations
    all_prompts_used = set()
    total_prompts_generated = 0
    
    # This would need to be tracked during the experiment
    # For now, we'll show the current prompt state
    current_prompts = experiment_state.get('prompts', [])
    prompt_mapping = experiment_state.get('prompt_to_int', {})
    
    print(f"Final generation prompt count: {len(current_prompts)}")
    print(f"Total unique prompts available: {len(prompt_mapping)}")
    
    if current_prompts:
        print("\nFinal generation prompts:")
        for i, prompt in enumerate(current_prompts):
            preview = prompt[:80] + "" if len(prompt) > 80 else prompt
            print(f"  {i}: {preview}")
    
    print("#"*50)


def main(cfg: Config):
    # Record experiment start time
    experiment_start_time = datetime.now()

    ####################################################################
    ##### Phase 0: Initialize basic experiment state (without CVT) #####
    ####################################################################
    print("MAIN: Initialize Experiment Variables")
    experiment_state = initialize_experiment(cfg)
    experiment_state['start_time'] = experiment_start_time
    
    print(f"\n MAIN: Experiment has been Initialised for Evolution: {cfg.GENERATIONS} Generations")
    
    #############################################################
    ##### Phase 1: Initialize Archive from Pregen Networks ######
    #############################################################
    print("\n" + "="*50)
    print("MAIN: Initialize and Populate Archive Using the Pre-Generated Networks")
    
    # Select Pregen directory based on the base architecture # TODO: Possible incompatibility of defined paths in the cfg file
    if cfg.NETWORK_TYPE == "single_label": 
        pregen_dir = cfg.PREGEN_DIR_SINGLE_LABEL
    elif cfg.NETWORK_TYPE == "image_regression": 
        pregen_dir = cfg.PREGEN_DIR_IMAGE_REGRESSION
    elif cfg.NETWORK_TYPE == "resnet": 
        pregen_dir = cfg.PREGEN_DIR_RESNET
    
    # Resolve relative paths to absolute paths relative to project root
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pregen_dir = os.path.join(project_root, pregen_dir)
    print(f"Resolved pregen directory to: {pregen_dir}")

    # Initialize archive using actual pregen networks with new descriptors
    success = init_archive_pregen(cfg, experiment_state, pregen_dir)
    
    if not success:
        print("MAIN: Failed to initialize archive from pregen networks. Exiting.")
        return None, None 
    
    print(f"MAIN: Archive successfully initialized with pregens: {len(experiment_state['nets_archive'])}/{cfg.N_NICHES} niches filled")

    #######################################
    ##### Phase 2: Evolutionary Phase #####
    ####################################### 
    print("\n" + "="*50)
    print(f"MAIN: Evolutionary Phase with {cfg.GENERATIONS} Generations")
    
    # Initialize temperature tracking for the evolutionary phase
    current_temperature = cfg.TEMPERATURE
    previous_best_fitness = float('inf')  # Track best fitness from previous generation
    
    for gen_i in range(cfg.GENERATIONS):

        # Set current generation in state for logging
        experiment_state['current_generation'] = gen_i 
        experiment_state['current_temperature'] = current_temperature  # Store current temperature in state
        
        # Generate new random prompts for this generation
        new_prompts = generate_new_random_prompts(experiment_state, cfg)
        
        logger = experiment_state.get('logger')
        if logger:
            phase = "Evolutionary"
            logger.log_generation_start(gen_i, phase)
        
        print(f"\n [Generation: {gen_i}] ")
        print(f"Current temperature: {current_temperature:.3f}")
        print(f"Using {len(new_prompts)} new random prompts for this generation")

        # Generate candidates 
        generated_candidates = generate_candidates(cfg, experiment_state)
        if not generated_candidates:
            print("No candidates were generated. Skipping generation.")
            if logger:
                logger.log_generation_end(gen_i, False, "No candidates were generated. Skipping generation.")
            continue

        # Validate candidates 
        valid_nets = validate_candidates(cfg, generated_candidates, experiment_state['path_nets'], gen_i, experiment_state)
        if not valid_nets:
            print("No valid networks survived validation. Skipping generation.")
            if logger:
                logger.log_generation_end(gen_i, False, "No valid networks survived validation. Skipping generation.")
            continue
            
        # Evaluate candidates
        fitness_results = evaluate_candidates(valid_nets, cfg, experiment_state['curios_net'], experiment_state)

        # Temperature adjustment based on performance
        if fitness_results:
            # Find the best fitness in current generation
            current_best_fitness = min(result['loss'] for result in fitness_results)
            
            # Adjust temperature based on performance comparison
            if current_best_fitness <= previous_best_fitness:
                # Performance improved or stayed the same - increase exploration
                current_temperature = min(current_temperature + 0.05, 1.0)
                print(f"Performance improved/stayed same (best: {current_best_fitness:.4f} vs prev: {previous_best_fitness:.4f}). Increasing temperature to {current_temperature:.3f}")
            else:
                # Performance worsened - decrease exploration
                current_temperature = max(current_temperature - 0.05, cfg.TEMP_MIN)
                print(f"Performance worsened (best: {current_best_fitness:.4f} vs prev: {previous_best_fitness:.4f}). Decreasing temperature to {current_temperature:.3f}")
            
            # Update previous best fitness for next generation
            previous_best_fitness = current_best_fitness
        else:
            print("No fitness results available. Keeping temperature unchanged.")

        # Updating the archives at the end of the generation
        update_archives(fitness_results, experiment_state, cfg, gen_i)
        
        # Log prompt diversity for this generation
        if logger:
            # Extract prompts used from the generated candidates
            prompts_used = []
            if generated_candidates:
                prompts_used = [candidate.get('prompt', '') for candidate in generated_candidates if candidate.get('prompt')]
            
            # Log prompt diversity
            logger.log_prompt_diversity(gen_i, prompts_used, experiment_state['prompt_to_int'])
            
            # Log successful generation completion
            archive_size = len(experiment_state['nets_archive'])
            coverage = (archive_size / cfg.N_NICHES) * 100
            logger.log_archive_status(gen_i, archive_size, cfg.N_NICHES, coverage)
            logger.log_generation_end(gen_i, True, f"Completed with {len(fitness_results)} evaluations")
        else:
            # Fallback logging if logger is not available
            print(f"Generation {gen_i} completed with {len(fitness_results)} evaluations")
            print(f"Archive size: {len(experiment_state['nets_archive'])}/{cfg.N_NICHES} niches filled")


    ######################################
    ##### Phase 3: Log final results #####
    ######################################
    # Save final experiment summary
    print("\n MAIN: Saving Final Experiment Summary")
    save_experiment_summary(experiment_state, cfg, experiment_state['path_nets'])
    update_experiment_summary(experiment_state, cfg, experiment_state['path_nets'])
    
    # Log final prompt diversity summary
    log_final_prompt_diversity_summary(experiment_state, cfg)
    
    # Log final experiment summary
    if logger:
        final_stats = {
            'total_generations': cfg.GENERATIONS,
            'final_archive_size': len(experiment_state['nets_archive']),
            'archive_coverage': f"{len(experiment_state['nets_archive'])}/{cfg.N_NICHES} ({len(experiment_state['nets_archive'])/cfg.N_NICHES*100:.1f}%)",
            'experiment_duration': f"{(datetime.now() - experiment_start_time).total_seconds():.1f} seconds",
            'mutation_model': cfg.MUTATION,
            'temperature': cfg.TEMPERATURE,
            'training_epochs': cfg.NET_TRAINING_EPOCHS
        }
        logger.log_experiment_summary(final_stats)
    
    experiment_state['log_file'].close()
    print("\nEvolution finished.")

    # Save archive results
    save_archive_results(experiment_state, cfg, experiment_state['path_nets'])
    
    ######################################
    ##### Phase 4: plot results #####
    ######################################
    # Create plots
    create_plots(experiment_state, experiment_state['path_nets'], cfg)
    
    return experiment_state['nets_archive'], experiment_state['prompt_archive']

if __name__ == "__main__":
    import argparse
    from conf.config import Config

    parser = argparse.ArgumentParser(description="Run the full NAS pipeline with custom configurations.")
    
    # Expose key Config variables as command-line arguments
    parser.add_argument('--task-type', type=str, default=Config.TASK_TYPE, help=f'Task type for the experiment (default: {Config.TASK_TYPE})')
    parser.add_argument('--network-type', type=str, default=Config.NETWORK_TYPE, help=f'Base network architecture (default: {Config.NETWORK_TYPE})')
    parser.add_argument('--generations', type=int, default=Config.GENERATIONS, help=f'Number of evolutionary generations to run (default: {Config.GENERATIONS})')
    parser.add_argument('--mutation', type=str, default=Config.MUTATION, help=f'LLM to use for mutation (default: {Config.MUTATION})')
    parser.add_argument('--net-training-epochs', type=int, default=Config.NET_TRAINING_EPOCHS, help=f'Number of epochs to train each network (default: {Config.NET_TRAINING_EPOCHS})')
    parser.add_argument('--n-niches', type=int, default=Config.N_NICHES, help=f'Number of niches in the MAP-Elites archive (default: {Config.N_NICHES})')

    args = parser.parse_args()

    # Create a Config instance, overriding defaults with any provided command-line arguments
    cfg = Config(
        TASK_TYPE=args.task_type,
        NETWORK_TYPE=args.network_type,
        GENERATIONS=args.generations,
        MUTATION=args.mutation,
        NET_TRAINING_EPOCHS=args.net_training_epochs,
        N_NICHES=args.n_niches
    )

    main(cfg)