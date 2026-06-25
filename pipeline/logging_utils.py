import os
import json
import csv
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import torch
from fvcore.nn import FlopCountAnalysis
from conf.config import Config

class ExperimentLogger:
    """
    logging system for tracking experiment progress, errors, and debugging information
    """
    
    def __init__(self, experiment_path: str, experiment_name: str):
        """
        Initialize the logger with experiment-specific paths
        """
        self.experiment_path = experiment_path
        self.experiment_name = experiment_name
        self.log_file_path = os.path.join(experiment_path, f"{experiment_name}_experiment.log")
        self.error_log_path = os.path.join(experiment_path, f"{experiment_name}_errors.log")
        self.generation_log_path = os.path.join(experiment_path, f"{experiment_name}_generations.log")
        
        # Create log files
        self._setup_log_files()
        
    def _setup_log_files(self):
        """Create and initialize log files with headers"""
        # Main experiment log
        with open(self.log_file_path, 'w') as f:
            f.write(f"=== EXPERIMENT LOG ===\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Log file: {self.log_file_path}\n")
            f.write("=" * 50 + "\n\n")
        
        # Error log
        with open(self.error_log_path, 'w') as f:
            f.write(f"=== ERROR LOG ===\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
        
        # Generation log
        with open(self.generation_log_path, 'w') as f:
            f.write(f"=== GENERATION LOG ===\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write("=" * 50 + "\n\n")
    
    def log(self, message: str, level: str = "INFO", generation: Optional[int] = None):
        """
        Log a message with timestamp and level
        """
        timestamp = datetime.now().isoformat()
        gen_prefix = f"[Gen {generation}] " if generation is not None else ""
        log_entry = f"[{timestamp}] {level}: {gen_prefix}{message}"
        
        # Write to main log file
        with open(self.log_file_path, 'a') as f:
            f.write(log_entry + "\n")
        
        # Also print to console for immediate feedback
        print(log_entry)
        
        # If it's an error, also write to error log
        if level == "ERROR":
            with open(self.error_log_path, 'a') as f:
                f.write(log_entry + "\n")
        
        # If generation is specified, write to generation log
        if generation is not None:
            with open(self.generation_log_path, 'a') as f:
                f.write(log_entry + "\n")
    
    def log_generation_start(self, generation: int, phase: str):
        """Log the start of a generation."""
        self.log(f"Starting generation {generation} - Phase: {phase}", "INFO", generation)
    
    def log_generation_end(self, generation: int, success: bool, reason: str = ""):
        """Log the end of a generation."""
        status = "SUCCESS" if success else "FAILED"
        message = f"Generation {generation} {status}"
        if reason:
            message += f"- Reason: {reason}"
        self.log(message, "INFO" if success else "WARNING", generation)
    
    def log_prompt_diversity(self, generation: int, prompts_used: list, prompt_mapping: dict):
        """
        Log prompt diversity information for a generation
        """
        if not prompts_used:
            return
            
        # Count unique prompts used
        unique_prompts = set(prompts_used)
        total_prompts = len(prompts_used)
        diversity_ratio = len(unique_prompts) / total_prompts if total_prompts > 0 else 0
        
        # Get prompt indices for logging
        prompt_indices = []
        for prompt in prompts_used:
            if prompt in prompt_mapping:
                prompt_indices.append(prompt_mapping[prompt])
            else:
                prompt_indices.append("Unknown")
        
        message = (f"Prompt diversity for generation {generation}: "
                  f"Total prompts: {total_prompts}, "
                  f"Unique prompts: {len(unique_prompts)}, "
                  f"Diversity ratio: {diversity_ratio:.2f}, "
                  f"Prompt indices: {prompt_indices}")
        
        self.log(message, "INFO", generation)
        
        # Save detailed prompt usage data
        prompt_usage_file = os.path.join(self.experiment_path, f"prompt_usage_gen_{generation}.json")
        prompt_data = {
            'generation': generation,
            'timestamp': datetime.now().isoformat(),
            'total_prompts': total_prompts,
            'unique_prompts': len(unique_prompts),
            'diversity_ratio': diversity_ratio,
            'prompts_used': [
                {
                    'prompt_text': prompt,
                    'prompt_index': prompt_mapping.get(prompt, "Unknown"),
                    'usage_count': prompts_used.count(prompt)
                }
                for prompt in unique_prompts
            ]
        }
        
        try:
            with open(prompt_usage_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)
        except Exception as e:
            self.log(f"Failed to save prompt usage data: {e}", "ERROR", generation)
    
    def log_candidate_generation(self, generation: int, total_candidates: int, successful_candidates: int, 
                                failed_reasons: List[str] = None):
        """Log candidate generation results."""
        success_rate = (successful_candidates / total_candidates * 100) if total_candidates > 0 else 0
        message = f"Candidate generation: {successful_candidates}/{total_candidates} successful ({success_rate:.1f}%)"
        self.log(message, "INFO", generation)
        
        if failed_reasons:
            for i, reason in enumerate(failed_reasons):
                self.log(f"  Failed candidate {i+1}: {reason}", "WARNING", generation)
    
    def log_validation_results(self, generation: int, total_candidates: int, valid_candidates: int, 
                              validation_failures: List[str] = None):
        """Log validation results."""
        validation_rate = (valid_candidates / total_candidates * 100) if total_candidates > 0 else 0
        message = f"Validation results: {valid_candidates}/{total_candidates} valid ({validation_rate:.1f}%)"
        self.log(message, "INFO", generation)
        
        if validation_failures:
            for i, failure in enumerate(validation_failures):
                self.log(f"  Validation failure {i+1}: {failure}", "WARNING", generation)
    
    def log_evaluation_results(self, generation: int, total_evaluated: int, successful_evaluations: int,
                              archive_additions: int):
        """Log evaluation results."""
        eval_rate = (successful_evaluations / total_evaluated * 100) if total_evaluated > 0 else 0
        addition_rate = (archive_additions / total_evaluated * 100) if total_evaluated > 0 else 0
        message = f"Evaluation results: {successful_evaluations}/{total_evaluated} successful ({eval_rate:.1f}%), {archive_additions} added to archive ({addition_rate:.1f}%)"
        self.log(message, "INFO", generation)
    
    def log_archive_status(self, generation: int, archive_size: int, total_niches: int, coverage_percentage: float):
        """Log archive status."""
        message = f"Archive status: {archive_size}/{total_niches} niches filled ({coverage_percentage:.1f}%)"
        self.log(message, "INFO", generation)
    
    def log_error(self, error: Exception, context: str = "", generation: Optional[int] = None):
        """Log an error with full traceback."""
        error_message = f"ERROR in {context}: {str(error)}"
        self.log(error_message, "ERROR", generation)
        
        # Also log the full traceback to error log
        import traceback
        with open(self.error_log_path, 'a') as f:
            f.write(f"[{datetime.now().isoformat()}] ERROR in {context}:\n")
            f.write(f"Exception: {str(error)}\n")
            f.write("Traceback:\n")
            f.write(traceback.format_exc())
            f.write("\n" + "="*50 + "\n\n")
    
    def log_experiment_summary(self, final_stats: Dict[str, Any]):
        """Log final experiment summary."""
        self.log("=== EXPERIMENT SUMMARY ===", "INFO")
        for key, value in final_stats.items():
            self.log(f"  {key}: {value}", "INFO")
        self.log("=== END EXPERIMENT ===", "INFO")

def setup_logging(experiment_path: str, experiment_name: str) -> ExperimentLogger:
    """
    Create and return a logger instance for the experiment
    """
    return ExperimentLogger(experiment_path, experiment_name)


def save_archive_map_data(state: dict, gen_i: int, save_path: str, cfg: Config):
    """
    Save archive map data for visualization and analysis.
    """
    archive_data = {
        'generation': int(gen_i),
        'timestamp': datetime.now().isoformat(),
        'archive_size': int(len(state['nets_archive'])),
        'prompt_archive_size': int(len(state['prompt_archive'])),
        'total_niches': int(len(state['kdt'].data)),
        'niche_coverage': float(len(state['nets_archive']) / len(state['kdt'].data)),
        'networks': []
    }
    
    # Get descriptor bounds for normalization
    bounds = state.get('descriptor_bounds', {
        'diversity_min': 0.0, 'diversity_max': 1.0, 
        'efficiency_min': 0.0, 'efficiency_max': 1.0
    })
    
    for niche, species in state['nets_archive'].items():
        # Handle different descriptor formats
        if len(species.desc) == 2:
            # New descriptor format: [diversity_score, efficiency_ratio]
            raw_diversity, raw_efficiency = species.desc
            
            # Calculate normalized descriptors using the same logic as archive_update
            d_range = bounds['diversity_max'] - bounds['diversity_min']
            e_range = bounds['efficiency_max'] - bounds['efficiency_min']
            
            norm_diversity = (raw_diversity - bounds['diversity_min']) / d_range if d_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
            norm_efficiency = (raw_efficiency - bounds['efficiency_min']) / e_range if e_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
            
            # Clamp to ensure they are strictly within [0, 1]
            norm_diversity = np.clip(norm_diversity, 0, 1)
            norm_efficiency = np.clip(norm_efficiency, 0, 1)
            
        else:
            # Old descriptor format: [param_count, flops]
            param_count, flops = species.desc
            
            # Calculate normalized descriptors
            log_flops = np.log10(flops)
            log_min_flops = np.log10(1e6)
            log_max_flops = np.log10(2e9)
            
            log_params = np.log10(param_count)
            log_min_params = np.log10(5e6)
            log_max_params = np.log10(5e7)
            
            norm_params = (log_params - log_min_params) / (log_max_params - log_min_params)
            norm_flops = (log_flops - log_min_flops) / (log_max_flops - log_min_flops)
            
            norm_params = np.clip(norm_params, 0, 1)
            norm_flops = np.clip(norm_flops, 0, 1)
            
            norm_diversity = norm_params
            norm_efficiency = norm_flops
            raw_diversity = param_count
            raw_efficiency = flops
        
        network_data = {
            'niche': [float(x) for x in niche], 
            'normalized_descriptors': [float(norm_diversity), float(norm_efficiency)],
            'raw_descriptors': [float(raw_diversity), float(raw_efficiency)],
            'fitness': float(species.fitness),
            'curiosity': float(species.curiosity),
            'network_path': str(species.net_path),
            'generation_added': int(gen_i)
        }
        archive_data['networks'].append(network_data)
    
    # Save as JSON for easy analysis
    archive_file = os.path.join(save_path, f"archive_map_gen_{gen_i}.json")
    with open(archive_file, 'w') as f:
        json.dump(archive_data, f, indent=2)
    
    # Save as CSV for plotting
    csv_file = os.path.join(save_path, f"archive_map_gen_{gen_i}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['niche_x', 'niche_y', 'norm_params', 'norm_flops', 'raw_params', 'raw_flops', 
                        'fitness', 'curiosity', 'network_path', 'generation'])
        for network in archive_data['networks']:
            writer.writerow([
                float(network['niche'][0]), float(network['niche'][1]),
                float(network['normalized_descriptors'][0]), float(network['normalized_descriptors'][1]),
                float(network['raw_descriptors'][0]), float(network['raw_descriptors'][1]),
                float(network['fitness']), float(network['curiosity']),
                str(network['network_path']), int(network['generation_added'])
            ])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_training_metrics(fitness_results: list, gen_i: int, save_path: str, cfg: Config):
    """
    Save detailed training metrics for each network evaluated.
    """
    metrics_file = os.path.join(save_path, f"training_metrics_gen_{gen_i}.csv")
    
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'network_path', 'prompt', 'temperature', 'generation', 'epoch',
            'train_loss', 'train_acc', 'val_loss', 'val_acc', 'inference_time',
            'final_loss', 'final_acc', 'param_count', 'flops', 'added_to_archive'
        ])
        
        for result in fitness_results:
            network_path = result['path']
            prompt = result['prompt']
            temp = result['temp']
            final_loss = result['loss']
            final_acc = result.get('accuracy', 0.0)
            inference_time = result.get('inference_time', 0.0)
            added_to_archive = result.get('added_to_archive', False)
            
            # Get training history
            training_history = result.get('training_history', {})
            epochs = training_history.get('epochs', [])
            train_losses = training_history.get('train_losses', [])
            train_accs = training_history.get('train_accs', [])
            val_losses = training_history.get('val_losses', [])
            val_accs = training_history.get('val_accs', [])
            
            # Get network characteristics
            net = result['net']
            try:
                device = next(net.parameters()).device
                dummy_inputs = torch.zeros((1, 3, 32, 32)).to(device)
                flops = FlopCountAnalysis(net, dummy_inputs).total()
                param_count = count_parameters(net)
            except:
                flops, param_count = 0, 0
            
            # Write training history for each epoch
            for i, epoch in enumerate(epochs):
                writer.writerow([
                    network_path, prompt, temp, gen_i, epoch,
                    train_losses[i] if i < len(train_losses) else 0.0,
                    train_accs[i] if i < len(train_accs) else 0.0,
                    val_losses[i] if i < len(val_losses) else 0.0,
                    val_accs[i] if i < len(val_accs) else 0.0,
                    inference_time if i == len(epochs) - 1 else 0.0,  # Only on final epoch
                    final_loss if i == len(epochs) - 1 else 0.0,      # Only on final epoch
                    final_acc if i == len(epochs) - 1 else 0.0,       # Only on final epoch
                    param_count if i == len(epochs) - 1 else 0,       # Only on final epoch
                    flops if i == len(epochs) - 1 else 0,             # Only on final epoch
                    added_to_archive if i == len(epochs) - 1 else False  # Only on final epoch
                ])

def save_pipeline_effectiveness(state: dict, fitness_results: list, gen_i: int, save_path: str, cfg: Config):
    """
    Save pipeline effectiveness metrics for research analysis.
    """
    effectiveness_file = os.path.join(save_path, f"pipeline_effectiveness_gen_{gen_i}.json")
    
    # Calculate effectiveness metrics
    total_candidates = len(fitness_results)
    valid_candidates = sum(1 for r in fitness_results if r.get('valid', True))
    added_to_archive = sum(1 for r in fitness_results if r.get('added_to_archive', False))
    
    # Archive diversity metrics
    if state['nets_archive']:
        fitnesses = [s.fitness for s in state['nets_archive'].values()]
        
        # Handle different descriptor formats
        if len(next(iter(state['nets_archive'].values())).desc) == 2:
            # New descriptor format: [diversity_score, efficiency_ratio]
            diversity_scores = [s.desc[0] for s in state['nets_archive'].values()]
            efficiency_ratios = [s.desc[1] for s in state['nets_archive'].values()]
            
            archive_metrics = {
                'fitness_stats': {
                    'min': float(min(fitnesses)),
                    'max': float(max(fitnesses)),
                    'mean': float(sum(fitnesses)/len(fitnesses)),
                    'std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                    'median': float(sorted(fitnesses)[len(fitnesses)//2])
                },
                'diversity_stats': {
                    'min': float(min(diversity_scores)),
                    'max': float(max(diversity_scores)),
                    'mean': float(sum(diversity_scores)/len(diversity_scores)),
                    'std': float((sum((x - sum(diversity_scores)/len(diversity_scores))**2 for x in diversity_scores) / len(diversity_scores))**0.5)
                },
                'efficiency_stats': {
                    'min': float(min(efficiency_ratios)),
                    'max': float(max(efficiency_ratios)),
                    'mean': float(sum(efficiency_ratios)/len(efficiency_ratios)),
                    'std': float((sum((x - sum(efficiency_ratios)/len(efficiency_ratios))**2 for x in efficiency_ratios) / len(efficiency_ratios))**0.5)
                }
            }
        else:
            # Old descriptor format: [param_count, flops]
            param_counts = [s.desc[0] for s in state['nets_archive'].values()]
            flops_counts = [s.desc[1] for s in state['nets_archive'].values()]
            
            archive_metrics = {
                'fitness_stats': {
                    'min': float(min(fitnesses)),
                    'max': float(max(fitnesses)),
                    'mean': float(sum(fitnesses)/len(fitnesses)),
                    'std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                    'median': float(sorted(fitnesses)[len(fitnesses)//2])
                },
                'param_stats': {
                    'min': float(min(param_counts)),
                    'max': float(max(param_counts)),
                    'mean': float(sum(param_counts)/len(param_counts)),
                    'std': float((sum((x - sum(param_counts)/len(param_counts))**2 for x in param_counts) / len(param_counts))**0.5)
                },
                'flops_stats': {
                    'min': float(min(flops_counts)),
                    'max': float(max(flops_counts)),
                    'mean': float(sum(flops_counts)/len(flops_counts)),
                    'std': float((sum((x - sum(flops_counts)/len(flops_counts))**2 for x in flops_counts) / len(flops_counts))**0.5)
                }
            }
    else:
        archive_metrics = {}
    
    effectiveness_data = {
        'generation': int(gen_i),
        'timestamp': datetime.now().isoformat(),
        'candidate_metrics': {
            'total_generated': int(total_candidates),
            'valid_candidates': int(valid_candidates),
            'validation_rate': float(valid_candidates / total_candidates if total_candidates > 0 else 0),
            'added_to_archive': int(added_to_archive),
            'archive_addition_rate': float(added_to_archive / total_candidates if total_candidates > 0 else 0)
        },
        'archive_metrics': archive_metrics,
        'archive_coverage': {
            'filled_niches': int(len(state['nets_archive'])),
            'total_niches': int(len(state['kdt'].data)),
            'coverage_percentage': float(len(state['nets_archive']) / len(state['kdt'].data) * 100)
        },
        'prompt_effectiveness': {
            'prompt_archive_size': int(len(state['prompt_archive'])),
            'avg_curiosity': float(sum(s.curiosity for s in state['prompt_archive'].values()) / len(state['prompt_archive']) if state['prompt_archive'] else 0)
        }
    }
    
    with open(effectiveness_file, 'w') as f:
        json.dump(effectiveness_data, f, indent=2)


def save_experiment_summary(state: dict, cfg: Config, save_path: str):
    """
    Save a experiment summary for the final report.
    """
    summary_file = os.path.join(save_path, "experiment_summary.json")
    
    # Calculate final metrics
    if state['nets_archive']:
        fitnesses = [s.fitness for s in state['nets_archive'].values()]
        
        # Handle different descriptor formats
        if len(next(iter(state['nets_archive'].values())).desc) == 2:
            # New descriptor format: [diversity_score, efficiency_ratio]
            diversity_scores = [s.desc[0] for s in state['nets_archive'].values()]
            efficiency_ratios = [s.desc[1] for s in state['nets_archive'].values()]
            
            best_network = min(state['nets_archive'].values(), key=lambda x: x.fitness)
            worst_network = max(state['nets_archive'].values(), key=lambda x: x.fitness)
            
            final_metrics = {
                'best_fitness': float(best_network.fitness),
                'worst_fitness': float(worst_network.fitness),
                'avg_fitness': float(sum(fitnesses)/len(fitnesses)),
                'fitness_std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                'best_network_path': str(best_network.net_path),
                'worst_network_path': str(worst_network.net_path),
                'diversity_range': [float(min(diversity_scores)), float(max(diversity_scores))],
                'efficiency_range': [float(min(efficiency_ratios)), float(max(efficiency_ratios))],
                'archive_diversity': float(len(set(fitnesses)) / len(fitnesses))  # Unique fitness values ratio
            }
        else:
            # Old descriptor format: [param_count, flops]
            param_counts = [s.desc[0] for s in state['nets_archive'].values()]
            flops_counts = [s.desc[1] for s in state['nets_archive'].values()]
            
            best_network = min(state['nets_archive'].values(), key=lambda x: x.fitness)
            worst_network = max(state['nets_archive'].values(), key=lambda x: x.fitness)
            
            final_metrics = {
                'best_fitness': float(best_network.fitness),
                'worst_fitness': float(worst_network.fitness),
                'avg_fitness': float(sum(fitnesses)/len(fitnesses)),
                'fitness_std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                'best_network_path': str(best_network.net_path),
                'worst_network_path': str(worst_network.net_path),
                'param_range': [float(min(param_counts)), float(max(param_counts))],
                'flops_range': [float(min(flops_counts)), float(max(flops_counts))],
                'archive_diversity': float(len(set(fitnesses)) / len(fitnesses))  # Unique fitness values ratio
            }
    else:
        final_metrics = {}
    
    summary = {
        'experiment_config': {
            'generations': int(cfg.GENERATIONS),
            'init_num_nets': int(cfg.INIT_NUM_NETS),
            'num_nets': int(cfg.NUM_NETS),
            'roll_outs': int(cfg.ROLL_OUTS),
            'random_init_nets': int(cfg.RANDOM_INIT_NETS),
            'training_epochs': int(cfg.NET_TRAINING_EPOCHS),
            'temperature': float(cfg.TEMPERATURE),
            'mutation_model': str(cfg.MUTATION),
            'niches': int(cfg.N_NICHES),
            'dim_map': int(cfg.DIM_MAP)
        },
        'final_results': {
            'total_networks_in_archive': int(len(state['nets_archive'])),
            'total_prompts_in_archive': int(len(state['prompt_archive'])),
            'archive_coverage_percentage': float(len(state['nets_archive']) / cfg.N_NICHES * 100),
            'experiment_duration': float((datetime.now() - state.get('start_time', datetime.now())).total_seconds()),
            'final_metrics': final_metrics
        },
        'archive_analysis': {
            'niche_distribution': {},
            'fitness_distribution': {},
            'descriptor_space_coverage': {}
        }
    }
    
    # Add niche distribution analysis
    if state['nets_archive']:
        niche_counts = {}
        for species in state['nets_archive'].values():
            niche_key = str(species.centroid)
            niche_counts[niche_key] = int(niche_counts.get(niche_key, 0) + 1)
        
        summary['archive_analysis']['niche_distribution'] = niche_counts
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

def save_prompt_archive_data(state: dict, gen_i: int, save_path: str, cfg: Config):
    """
    Save prompt archive data with prompt text, integer mapping, temperature, and fitness.
    """
    prompt_data = {
        'generation': int(gen_i),
        'timestamp': datetime.now().isoformat(),
        'prompt_archive_size': int(len(state['prompt_archive'])),
        'prompts': []
    }
    
    for prompt_key, prompt_species in state['prompt_archive'].items():
        # Extract prompt integer and temperature from descriptors
        prompt_int = prompt_species.desc[0]
        temperature = prompt_species.desc[1]
        
        # Get the actual prompt text from the integer mapping
        if 'int_to_prompt' in state and 'prompts' in state:
            prompt_index = int(prompt_int * (len(state['prompts']) - 1))
            if 0 <= prompt_index < len(state['prompts']):
                prompt_text = state['prompts'][prompt_index]
            else:
                prompt_text = f"Unknown prompt (index: {prompt_index})"
        else:
            prompt_text = f"Prompt {prompt_int:.3f}"
        
        prompt_entry = {
            'prompt_key': str(prompt_key),
            'prompt_integer': float(prompt_int),
            'prompt_text': prompt_text,
            'temperature': float(temperature),
            'fitness': float(prompt_species.fitness),
            'curiosity_score': float(prompt_species.curiosity),
            'timestamp': datetime.now().isoformat()
        }
        prompt_data['prompts'].append(prompt_entry)
    
    # Save as JSON
    prompt_file = os.path.join(save_path, f"prompt_archive_gen_{gen_i}.json")
    with open(prompt_file, 'w') as f:
        json.dump(prompt_data, f, indent=2)
    
    # Save as CSV for easy analysis
    csv_file = os.path.join(save_path, f"prompt_archive_gen_{gen_i}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt_integer', 'temperature', 'fitness', 'curiosity_score', 'prompt_text'])
        for prompt in prompt_data['prompts']:
            writer.writerow([
                prompt['prompt_integer'],
                prompt['temperature'],
                prompt['fitness'],
                prompt['curiosity_score'],
                prompt['prompt_text']
            ])

def save_parent_child_relationships(state: dict, gen_i: int, save_path: str, cfg: Config):
    """
    Track and save parent-child relationships for evolutionary analysis.
    """
    # Create parent-child relationships directory
    relationships_dir = os.path.join(save_path, "parent_child_relationships")
    os.makedirs(relationships_dir, exist_ok=True)
    
    # Get current generation's parent-child data from state
    generation_relationships = state.get('generation_relationships', [])
    
    relationships_data = {
        'generation': int(gen_i),
        'timestamp': datetime.now().isoformat(),
        'total_relationships': len(generation_relationships),
        'relationships': []
    }
    
    for rel in generation_relationships:
        relationship_entry = {
            'child_network_path': rel.get('child_path', ''),
            'parent_network_path': rel.get('parent_path', ''),
            'parent_prompt': rel.get('parent_prompt', ''),
            'parent_temperature': rel.get('parent_temperature', 0.0),
            'operation_type': rel.get('operation_type', 'unknown'),
            'child_fitness': rel.get('child_fitness', float('inf')),
            'parent_fitness': rel.get('parent_fitness', float('inf')),
            'success': rel.get('success', False),
            'generation': int(gen_i)
        }
        relationships_data['relationships'].append(relationship_entry)
    
    # Save generation-specific relationships
    gen_file = os.path.join(relationships_dir, f"parent_child_relationships_gen_{gen_i}.json")
    with open(gen_file, 'w') as f:
        json.dump(relationships_data, f, indent=2)
    
    # Update cumulative relationships file
    cumulative_file = os.path.join(relationships_dir, "parent_child_relationships_cumulative.json")
    cumulative_data = []
    
    if os.path.exists(cumulative_file):
        with open(cumulative_file, 'r') as f:
            cumulative_data = json.load(f)
    
    cumulative_data.extend(relationships_data['relationships'])
    
    with open(cumulative_file, 'w') as f:
        json.dump(cumulative_data, f, indent=2)
    
    # Save CSV for analysis
    csv_file = os.path.join(relationships_dir, f"parent_child_relationships_gen_{gen_i}.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'child_network', 'parent_network', 'parent_prompt', 
                        'parent_temperature', 'operation_type', 'child_fitness', 'parent_fitness', 'success'])
        for rel in relationships_data['relationships']:
            writer.writerow([
                rel['generation'],
                os.path.basename(rel['child_network_path']),
                os.path.basename(rel['parent_network_path']),
                rel['parent_prompt'][:50] + '' if len(rel['parent_prompt']) > 50 else rel['parent_prompt'],
                rel['parent_temperature'],
                rel['operation_type'],
                rel['child_fitness'],
                rel['parent_fitness'],
                rel['success']
            ])
    
    # Save cumulative CSV
    cumulative_csv_file = os.path.join(relationships_dir, "parent_child_relationships_cumulative.csv")
    with open(cumulative_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'child_network', 'parent_network', 'parent_prompt', 
                        'parent_temperature', 'operation_type', 'child_fitness', 'parent_fitness', 'success'])
        for rel in cumulative_data:
            writer.writerow([
                rel['generation'],
                os.path.basename(rel['child_network_path']),
                os.path.basename(rel['parent_network_path']),
                rel['parent_prompt'][:50] + '' if len(rel['parent_prompt']) > 50 else rel['parent_prompt'],
                rel['parent_temperature'],
                rel['operation_type'],
                rel['child_fitness'],
                rel['parent_fitness'],
                rel['success']
            ])

def save_all_networks_data(fitness_results: List[dict], gen_i: int, save_path: str, cfg: Config, state: dict):
    """
    Save data for ALL networks generated in this generation    """
    if not fitness_results:
        return
    
    # Calculate descriptors for all networks
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    bounds = state.get('descriptor_bounds', {
        'diversity_min': 0.0, 'diversity_max': 1.0, 
        'efficiency_min': 0.0, 'efficiency_max': 1.0
    })
    
    all_networks_data = []
    
    for i, result in enumerate(fitness_results):
        try:
            net = result["net"].to(device)
            
            # Calculate descriptors using the same functions as archive_update
            from descriptors.architectural_diversity import architectural_diversity_score
            from descriptors.Efficiency_ratio import efficiency_ratio
            
            # Calculate raw descriptors
            try:
                raw_diversity = architectural_diversity_score(net, cfg)
                raw_efficiency = efficiency_ratio(net, cfg)
            except Exception as e:
                print(f"Warning: Could not calculate descriptors for network {i}: {e}")
                raw_diversity = cfg.DEFAULT_DESCRIPTOR_VALUE
                raw_efficiency = cfg.DEFAULT_DESCRIPTOR_VALUE
            
            # Calculate normalized descriptors
            d_range = bounds['diversity_max'] - bounds['diversity_min']
            e_range = bounds['efficiency_max'] - bounds['efficiency_min']
            
            norm_diversity = (raw_diversity - bounds['diversity_min']) / d_range if d_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
            norm_efficiency = (raw_efficiency - bounds['efficiency_min']) / e_range if e_range > 0 else cfg.DEFAULT_NORMALIZED_VALUE
            
            # Clamp to [0,1] range
            norm_diversity = np.clip(norm_diversity, 0, 1)
            norm_efficiency = np.clip(norm_efficiency, 0, 1)
            
            # Compile network data
            network_data = {
                'generation': gen_i,
                'network_id': i,
                'network_path': result.get('path', ''),
                'fitness_loss': float(result.get('loss', float('inf'))),
                'accuracy': float(result.get('accuracy', 0.0)) if 'accuracy' in result else None,
                'inference_time': float(result.get('inference_time', 0.0)),
                'raw_diversity': float(raw_diversity),
                'raw_efficiency': float(raw_efficiency),
                'normalized_diversity': float(norm_diversity),
                'normalized_efficiency': float(norm_efficiency),
                'added_to_archive': bool(result.get('added_to_archive', False)),
                'valid': bool(result.get('valid', True)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add parent information if available
            if 'parent_info' in result:
                parent_info = result['parent_info']
                network_data.update({
                    'operation_type': parent_info.get('operation_type', 'unknown'),
                    'parent_path': parent_info.get('parent_path', ''),
                    'parent_fitness': float(parent_info.get('parent_fitness', 0.0)),
                    'parent_prompt': parent_info.get('parent_prompt', ''),
                    'parent_temperature': float(parent_info.get('parent_temperature', 0.0))
                })
            else:
                network_data.update({
                    'operation_type': 'unknown',
                    'parent_path': '',
                    'parent_fitness': 0.0,
                    'parent_prompt': '',
                    'parent_temperature': 0.0
                })
            
            # Add training metrics if available
            if 'training_results' in result and result['training_results']:
                training_results = result['training_results']
                network_data.update({
                    'final_train_loss': float(training_results.get('final_train_loss', 0.0)),
                    'final_val_loss': float(training_results.get('final_val_loss', 0.0)),
                    'final_train_acc': float(training_results.get('final_train_acc', 0.0)) if 'final_train_acc' in training_results else None,
                    'final_val_acc': float(training_results.get('final_val_acc', 0.0)) if 'final_val_acc' in training_results else None,
                    'training_time': float(training_results.get('training_time', 0.0))
                })
            
            all_networks_data.append(network_data)
            
        except Exception as e:
            print(f"Warning: Could not process network {i} for logging: {e}")
            continue
    
    # Save as JSON
    json_file = os.path.join(save_path, f"all_networks_gen_{gen_i}.json")
    with open(json_file, 'w') as f:
        json.dump({
            'generation': gen_i,
            'timestamp': datetime.now().isoformat(),
            'total_networks': len(all_networks_data),
            'networks_added_to_archive': sum(1 for n in all_networks_data if n['added_to_archive']),
            'descriptor_bounds': bounds,
            'networks': all_networks_data
        }, f, indent=2)
    
    # Save as CSV for easy analysis
    csv_file = os.path.join(save_path, f"all_networks_gen_{gen_i}.csv")
    if all_networks_data:
        # Get all possible fieldnames from the data
        fieldnames = set()
        for network in all_networks_data:
            fieldnames.update(network.keys())
        fieldnames = sorted(list(fieldnames))
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for network in all_networks_data:
                # Fill missing fields with None/empty values
                row = {field: network.get(field, '') for field in fieldnames}
                writer.writerow(row)
    
    print(f"Network data saved: {len(all_networks_data)} networks to {json_file} and {csv_file}")


def save_evolutionary_operator_stats(state: dict, gen_i: int, save_path: str, cfg: Config):
    """
    Track and save mutation/crossover success rates and statistics.
    """
    # Create evolutionary operator stats directory
    operator_stats_dir = os.path.join(save_path, "evolutionary_operator_stats")
    os.makedirs(operator_stats_dir, exist_ok=True)
    
    # Get current generation's operator data from state
    generation_operator_data = state.get('generation_operator_data', {})
    
    operator_stats = {
        'generation': int(gen_i),
        'timestamp': datetime.now().isoformat(),
        'operator_usage': generation_operator_data.get('operator_usage', {}),
        'success_rates': generation_operator_data.get('success_rates', {}),
        'fitness_improvements': generation_operator_data.get('fitness_improvements', {})
    }
    
    # Save generation-specific stats
    gen_file = os.path.join(operator_stats_dir, f"evolutionary_operator_stats_gen_{gen_i}.json")
    with open(gen_file, 'w') as f:
        json.dump(operator_stats, f, indent=2)
    
    # Update cumulative stats
    cumulative_file = os.path.join(operator_stats_dir, "evolutionary_operator_stats_cumulative.json")
    cumulative_data = []
    
    if os.path.exists(cumulative_file):
        with open(cumulative_file, 'r') as f:
            cumulative_data = json.load(f)
    
    cumulative_data.append(operator_stats)
    
    with open(cumulative_file, 'w') as f:
        json.dump(cumulative_data, f, indent=2)
    
    # Calculate and save summary statistics
    if cumulative_data:
        summary_stats = {
            'total_generations': len(cumulative_data),
            'mutation_stats': {
                'total_usage': sum(gen.get('operator_usage', {}).get('mutation', 0) for gen in cumulative_data),
                'total_success': sum(gen.get('success_rates', {}).get('mutation', {}).get('successful', 0) for gen in cumulative_data),
                'success_rate': 0.0
            },
            'crossover_stats': {
                'total_usage': sum(gen.get('operator_usage', {}).get('crossover', 0) for gen in cumulative_data),
                'total_success': sum(gen.get('success_rates', {}).get('crossover', {}).get('successful', 0) for gen in cumulative_data),
                'success_rate': 0.0
            }
        }
        
        # Calculate success rates
        if summary_stats['mutation_stats']['total_usage'] > 0:
            summary_stats['mutation_stats']['success_rate'] = (
                summary_stats['mutation_stats']['total_success'] / 
                summary_stats['mutation_stats']['total_usage']
            )
        
        if summary_stats['crossover_stats']['total_usage'] > 0:
            summary_stats['crossover_stats']['success_rate'] = (
                summary_stats['crossover_stats']['total_success'] / 
                summary_stats['crossover_stats']['total_usage']
            )
        
        summary_file = os.path.join(operator_stats_dir, "evolutionary_operator_stats_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)

def update_experiment_summary(state: dict, cfg: Config, save_path: str):
    """
    Save a experiment summary including all config parameters.
    """
    summary_file = os.path.join(save_path, "experiment_summary.json")
    
    # Calculate final metrics
    if state['nets_archive']:
        fitnesses = [s.fitness for s in state['nets_archive'].values()]
        
        # Handle different descriptor formats
        if len(next(iter(state['nets_archive'].values())).desc) == 2:
            # New descriptor format: [diversity_score, efficiency_ratio]
            diversity_scores = [s.desc[0] for s in state['nets_archive'].values()]
            efficiency_ratios = [s.desc[1] for s in state['nets_archive'].values()]
            
            best_network = min(state['nets_archive'].values(), key=lambda x: x.fitness)
            worst_network = max(state['nets_archive'].values(), key=lambda x: x.fitness)
            
            final_metrics = {
                'best_fitness': float(best_network.fitness),
                'worst_fitness': float(worst_network.fitness),
                'avg_fitness': float(sum(fitnesses)/len(fitnesses)),
                'fitness_std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                'best_network_path': str(best_network.net_path),
                'worst_network_path': str(worst_network.net_path),
                'diversity_range': [float(min(diversity_scores)), float(max(diversity_scores))],
                'efficiency_range': [float(min(efficiency_ratios)), float(max(efficiency_ratios))],
                'archive_diversity': float(len(set(fitnesses)) / len(fitnesses))
            }
        else:
            # Old descriptor format: [param_count, flops]
            param_counts = [s.desc[0] for s in state['nets_archive'].values()]
            flops_counts = [s.desc[1] for s in state['nets_archive'].values()]
            
            best_network = min(state['nets_archive'].values(), key=lambda x: x.fitness)
            worst_network = max(state['nets_archive'].values(), key=lambda x: x.fitness)
            
            final_metrics = {
                'best_fitness': float(best_network.fitness),
                'worst_fitness': float(worst_network.fitness),
                'avg_fitness': float(sum(fitnesses)/len(fitnesses)),
                'fitness_std': float((sum((x - sum(fitnesses)/len(fitnesses))**2 for x in fitnesses) / len(fitnesses))**0.5),
                'best_network_path': str(best_network.net_path),
                'worst_network_path': str(worst_network.net_path),
                'param_range': [float(min(param_counts)), float(max(param_counts))],
                'flops_range': [float(min(flops_counts)), float(max(flops_counts))],
                'archive_diversity': float(len(set(fitnesses)) / len(fitnesses))
            }
    else:
        final_metrics = {}
    
    # config parameters
    config_params = {
        'experiment_config': {
            'generations': int(cfg.GENERATIONS),
            'init_num_nets': int(cfg.INIT_NUM_NETS),
            'num_nets': int(cfg.NUM_NETS),
            'roll_outs': int(cfg.ROLL_OUTS),
            'random_init_nets': int(cfg.RANDOM_INIT_NETS),
            'training_epochs': int(cfg.NET_TRAINING_EPOCHS),
            'temperature': float(cfg.TEMPERATURE),
            'temp_min': float(cfg.TEMP_MIN),
            'temp_max': float(cfg.TEMP_MAX),
            'mutation_model': str(cfg.MUTATION),
            'niches': int(cfg.N_NICHES),
            'dim_map': int(cfg.DIM_MAP),
            'network_type': str(cfg.NETWORK_TYPE),
            'device': str(cfg.DEVICE),
            'batch_size': int(cfg.BATCH_SIZE),
            'learning_rate': float(cfg.LEARNING_RATE),
            'momentum': float(cfg.MOMENTUM),
            'num_classes': int(cfg.NUM_CLASSES),
            'img_channels': int(cfg.IMG_CHANNELS),
            'num_layers': int(cfg.NUM_LAYERS),
            'output_size': int(cfg.OUTPUT_SIZE) if cfg.OUTPUT_SIZE is not None else None,
            'successful_curiosity_bonus': float(cfg.SUCCESSFUL_CURIOSITY_BONUS),
            'base_curiosity_score': float(cfg.BASE_CURIOSITY_SCORE),
            'bounds_padding_percent': float(cfg.BOUNDS_PADDING_PERCENT),
            'default_descriptor_value': float(cfg.DEFAULT_DESCRIPTOR_VALUE),
            'default_normalized_value': float(cfg.DEFAULT_NORMALIZED_VALUE),
            'pregen_dir_single_label': str(cfg.PREGEN_DIR_SINGLE_LABEL),
            'pregen_dir_image_regression': str(cfg.PREGEN_DIR_IMAGE_REGRESSION),
            'pregen_dir_resnet': str(cfg.PREGEN_DIR_RESNET),
            'IMG_WIDTH': int(cfg.IMG_WIDTH),
            'IMG_HEIGHT': int(cfg.IMG_HEIGHT)
        },
        'final_results': {
            'total_networks_in_archive': int(len(state['nets_archive'])),
            'total_prompts_in_archive': int(len(state['prompt_archive'])),
            'archive_coverage_percentage': float(len(state['nets_archive']) / cfg.N_NICHES * 100),
            'experiment_duration': float((datetime.now() - state.get('start_time', datetime.now())).total_seconds()),
            'final_metrics': final_metrics
        },
        'archive_analysis': {
            'niche_distribution': {},
            'fitness_distribution': {},
            'descriptor_space_coverage': {}
        }
    }
    
    # Add niche distribution analysis
    if state['nets_archive']:
        niche_counts = {}
        for species in state['nets_archive'].values():
            niche_key = str(species.centroid)
            niche_counts[niche_key] = int(niche_counts.get(niche_key, 0) + 1)
        
        config_params['archive_analysis']['niche_distribution'] = niche_counts
    
    with open(summary_file, 'w') as f:
        json.dump(config_params, f, indent=2)