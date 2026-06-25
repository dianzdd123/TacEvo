import torch
from descriptors.architectural_diversity import architectural_diversity_score
from descriptors.Efficiency_ratio import efficiency_ratio
from conf.config import Config
import numpy as np
from pipeline.archive_utils import to_specie, __add_to_archive, save_archive_to_path
from pipeline.logging_utils import save_archive_map_data, save_training_metrics, save_pipeline_effectiveness, save_prompt_archive_data, save_parent_child_relationships, save_evolutionary_operator_stats, save_all_networks_data
from typing import Tuple, Dict, List
import traceback

def update_archives(fitness_results: List[dict], state: dict, cfg: Config, gen_i: int):
    """
    Updates network and prompt archives using a descriptor-based QD approach
    Calculates descriptors, normalizes them, and adds them to the archive
    """

    print("##### Updating Archives Descriptors #####")
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"

    # 1. Initialize descriptor bounds on the first run
    if 'descriptor_bounds' not in state or state['descriptor_bounds'] is None:
        state['descriptor_bounds'] = _initialize_descriptor_bounds(fitness_results, cfg, device)
    
    bounds = state['descriptor_bounds']
    print(f"Archive size before update: {len(state['nets_archive'])} networks")

    # 2. Process each evaluated network from the generation
    for result in fitness_results:
        net = result["net"].to(device)
        
        # Calculate raw and normalized descriptors for the network
        raw_descriptors = _calculate_descriptors(net, cfg)
        normalized_descriptors = _normalize_descriptors(raw_descriptors, bounds, cfg)
        
        print(f"Network descriptors: Div={raw_descriptors[0]:.3f}, Eff={raw_descriptors[1]:.3f} | norm: {normalized_descriptors}")

        # Update Network Archive
        s_net = to_specie(net, result["loss"], np.array(raw_descriptors), result["path"])
        net_added = __add_to_archive(s_net, normalized_descriptors, state['nets_archive'], state['kdt'], "network")
        result["added_to_archive"] = net_added

        if net_added:
            state['curios_net'] = net
            print(f"SUCCESS: Network added/updated in archive (New Total: {len(state['nets_archive'])})")
            
            # Update operator success tracking if this network was generated this generation
            if 'parent_info' in result:
                operation_type = result['parent_info']['operation_type']
                state['generation_operator_data']['success_rates'][operation_type]['successful'] += 1
                
                # Update relationship success status
                for rel in state.get('generation_relationships', []):
                    if rel['child_path'] == result["path"]:
                        rel['success'] = True
                        break
        
        # Update Prompt Archive (if not a crossover operation)
        operation_type = result.get("operation_type", "unknown")
        if operation_type != 'crossover':
            if net_added:
                curiosity_score = cfg.SUCCESSFUL_CURIOSITY_BONUS + (1.0 / (1.0 + result["loss"]))
            else:
                curiosity_score = cfg.BASE_CURIOSITY_SCORE

            prompt_fitness = 1.0 if net_added else 0.0
            
            # hardcoded normalization for prompt descriptors
            norm_prompt = state['prompt_to_int'].get(result["prompt"], 0.5) / (len(state['prompts']) - 1) if state['prompts'] else 0.5
            norm_temp = result.get("temp", 0.5)
            prompt_descriptors = np.array([norm_prompt, norm_temp])
            
            s_prompt = to_specie(net, prompt_fitness, prompt_descriptors, result["path"])
            s_prompt.curiosity = curiosity_score
            
            __add_to_archive(s_prompt, prompt_descriptors, state['prompt_archive'], state['kdt'], "prompt")
            print(f"SUCCESS: Processed prompt (curiosity: {curiosity_score:.3f})")

    print(f"Archive size after update: {len(state['nets_archive'])} networks")

    # 3. Save logs, data, and plots for the generation
    _log_and_save_generation_data(fitness_results, state, cfg, gen_i)

def _calculate_descriptors(net: torch.nn.Module, cfg: Config) -> Tuple[float, float]:
    """Calculates the raw diversity and efficiency scores for a single network."""
    try:
        diversity_score = architectural_diversity_score(net, cfg)
        efficiency_score = efficiency_ratio(net, cfg)
        return diversity_score, efficiency_score
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not calculate descriptors. Error: {e}")
        traceback.print_exc()
        return cfg.DEFAULT_DESCRIPTOR_VALUE, cfg.DEFAULT_DESCRIPTOR_VALUE


def _initialize_descriptor_bounds(fitness_results: List[dict], cfg: Config, device: str) -> Dict[str, float]:
    """Calculates the adaptive bounds for descriptors from the initial population."""
    print("Calculating adaptive bounds from current networks")
    all_diversity_scores = []
    all_efficiency_ratios = []
    
    for result in fitness_results:
        net = result["net"].to(device)
        diversity, efficiency = _calculate_descriptors(net, cfg)
        all_diversity_scores.append(diversity)
        all_efficiency_ratios.append(efficiency)

    if not all_diversity_scores or not all_efficiency_ratios:
        return {'diversity_min': 0.0, 'diversity_max': 1.0, 'efficiency_min': 0.0, 'efficiency_max': 1.0}

    # Calculate bounds with padding
    d_min, d_max = min(all_diversity_scores), max(all_diversity_scores)
    e_min, e_max = min(all_efficiency_ratios), max(all_efficiency_ratios)
    
    d_range = d_max - d_min
    e_range = e_max - e_min
    
    bounds = {
        'diversity_min': max(0.0, d_min - cfg.BOUNDS_PADDING_PERCENT * d_range),
        'diversity_max': min(1.0, d_max + cfg.BOUNDS_PADDING_PERCENT * d_range),
        'efficiency_min': max(0.0, e_min - cfg.BOUNDS_PADDING_PERCENT * e_range),
        'efficiency_max': min(1.0, e_max + cfg.BOUNDS_PADDING_PERCENT * e_range)
    }
    
    print("Adaptive bounds calculated:")
    print(f"Diversity: [{bounds['diversity_min']:.3f}, {bounds['diversity_max']:.3f}]")
    print(f"Efficiency: [{bounds['efficiency_min']:.3f}, {bounds['efficiency_max']:.3f}]")
    return bounds

def _normalize_descriptors(raw_desc: Tuple[float, float], bounds: Dict[str, float], cfg: Config) -> np.ndarray:
    """Normalizes raw descriptor scores to the [0, 1] range."""
    raw_diversity, raw_efficiency = raw_desc
    
    d_range = bounds['diversity_max'] - bounds['diversity_min']
    e_range = bounds['efficiency_max'] - bounds['efficiency_min']
    
    norm_diversity = (raw_diversity - bounds['diversity_min']) / d_range if d_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
    norm_efficiency = (raw_efficiency - bounds['efficiency_min']) / e_range if e_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
    
    # Clamp to ensure they are strictly within [0, 1]
    norm_diversity = np.clip(norm_diversity, 0, 1)
    norm_efficiency = np.clip(norm_efficiency, 0, 1)
    
    return np.array([norm_diversity, norm_efficiency])

def _log_and_save_generation_data(fitness_results: List[dict], state: dict, cfg: Config, gen_i: int):
    """Saves archives, logs statistics, and updates plots for the generation."""
    print("Saving archives and logging generation data")

    # Save archives
    save_archive_to_path(state['nets_archive'], gen_i, "net", state['path_nets'])
    save_archive_to_path(state['prompt_archive'], gen_i, "prompt", state['path_nets'])
    
    # Log fitness statistics
    if state.get('log_file') and state['nets_archive']:
        fit_list = [float(x.fitness) for x in state['nets_archive'].values()]
        log_line = (f"{gen_i} {len(state['nets_archive'])} {len(state['prompt_archive'])} "
                    f"{min(fit_list):.6f} {sum(fit_list)/len(fit_list):.6f} {sorted(fit_list)[len(fit_list)//2]:.6f} "
                    f"{sorted(fit_list)[int(len(fit_list)*0.05)]:.6f} {sorted(fit_list)[int(len(fit_list)*0.95)]:.6f}\n")
        state['log_file'].write(log_line)
        state['log_file'].flush()
    
    # research logging and plotting
    save_archive_map_data(state, gen_i, state['path_nets'], cfg)
    save_training_metrics(fitness_results, gen_i, state['path_nets'], cfg)
    save_pipeline_effectiveness(state, fitness_results, gen_i, state['path_nets'], cfg)
    save_prompt_archive_data(state, gen_i, state['path_nets'], cfg)
    save_parent_child_relationships(state, gen_i, state['path_nets'], cfg)
    save_evolutionary_operator_stats(state, gen_i, state['path_nets'], cfg)
    # Save data for ALL networks (not just those in archive)
    save_all_networks_data(fitness_results, gen_i, state['path_nets'], cfg, state)
    _update_live_plots_new_descriptors(state, gen_i, fitness_results)


def _update_live_plots_new_descriptors(state: dict, gen_i: int, fitness_results: list): 
    """
    Update live plots with new descriptor methodology (Diversity + Efficiency).
    """
    print("  Updating live plots")
    
    if 'live_plotter' not in state:
        return
        
    try:
        # Get archive data for plotting
        archive_data = {
            'networks': [],
            'total_niches': len(state['kdt'].data) if state['kdt'] else 12
        }
        
        for species in state['nets_archive'].values():
            diversity_score, efficiency_score = species.desc
            archive_data['networks'].append({
                'raw_descriptors': [diversity_score, efficiency_score],
                'fitness': species.fitness,
                'curiosity': species.curiosity
            })
        
        state['live_plotter'].update_archive_evolution_plot(gen_i, archive_data)
        
        # Update training progress plot
        if state['nets_archive']:
            fit_list = [float(x.fitness) for x in state['nets_archive'].values()]
            state['live_plotter'].update_training_progress_plot(
                gen_i, len(state['nets_archive']), len(state['prompt_archive']),
                min(fit_list), sum(fit_list)/len(fit_list), sorted(fit_list)[len(fit_list)//2],
                sorted(fit_list)[int(len(fit_list)*0.05)], sorted(fit_list)[int(len(fit_list)*0.95)]
            )
        
        # Update pipeline effectiveness plot
        total_candidates = len(fitness_results)
        valid_candidates = sum(1 for r in fitness_results if r.get('valid', True))
        added_to_archive = sum(1 for r in fitness_results if r.get('added_to_archive', False))
        success_rate = (valid_candidates / total_candidates * 100) if total_candidates > 0 else 0
        archive_coverage = (len(state['nets_archive']) / len(state['kdt'].data) * 100) if state['kdt'] else 0
        
        state['live_plotter'].update_pipeline_effectiveness_plot(
            gen_i, total_candidates, valid_candidates, added_to_archive, 
            success_rate, archive_coverage
        )
        
    except Exception as e:
        print(f"Warning: Live plotting failed: {e}")