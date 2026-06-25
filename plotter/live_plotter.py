import os
import numpy as np
import matplotlib.pyplot as plt
from conf.config import Config

class LivePlotter:
    """
    Real-time plotting functionality for monitoring NAS evolution.
    """
    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.plots_dir = os.path.join(experiment_path, "live_plots")
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize matplotlib for non-interactive backend
        plt.ioff()  # Turn off interactive mode for better performance
        
        # Store data for cumulative plots
        self.archive_evolution_data = []
        self.training_progress_data = []
        self.pipeline_effectiveness_data = []
        
    def update_archive_evolution_plot(self, generation, archive_data):
        """
        Update and save the archive evolution plot in real-time.
        """
        try:
            # Add new data
            self.archive_evolution_data.append({
                'generation': generation,
                'networks': archive_data['networks'],
                'total_niches': archive_data['total_niches']
            })
            
            # Create evolution plot
            fig, axes = plt.subplots(1, min(3, len(self.archive_evolution_data)), 
                                   figsize=(6*min(3, len(self.archive_evolution_data)), 5))
            if len(self.archive_evolution_data) == 1:
                axes = [axes]
            
            # Show last 3 generations (or all if less than 3)
            recent_data = self.archive_evolution_data[-3:]
            
            for i, data in enumerate(recent_data):
                networks = data['networks']
                if not networks:
                    continue
                
                # Extract data
                diversity_scores = [n['raw_descriptors'][0] for n in networks]  # Architectural diversity (0-1)
                efficiency_ratios = [n['raw_descriptors'][1] for n in networks]  # Efficiency ratio
                fitnesses = [n['fitness'] for n in networks]
                
                # Create scatter plot
                scatter = axes[i].scatter(diversity_scores, efficiency_ratios, c=fitnesses, 
                                        cmap='viridis', s=100, alpha=0.8, edgecolors='black')
                
                axes[i].set_xlabel('Architectural Diversity Score')
                axes[i].set_ylabel('Efficiency Ratio')
                axes[i].set_title(f'Gen {data["generation"]}\n({len(networks)} networks)')
                axes[i].grid(True, alpha=0.3)
                
            
                cbar = plt.colorbar(scatter, ax=axes[i])
                cbar.set_label('Fitness (Validation Loss)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'live_archive_evolution.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not update archive evolution plot: {e}")
    
    def update_training_progress_plot(self, generation, archive_size, prompt_archive_size, 
                                    min_fitness, mean_fitness, median_fitness, p5_fitness, p95_fitness):
        """
        Update and save the training progress plot in real-time.
        """
        try:
            # Add new data
            self.training_progress_data.append({
                'generation': generation,
                'archive_size': archive_size,
                'prompt_archive_size': prompt_archive_size,
                'min_fitness': min_fitness,
                'mean_fitness': mean_fitness,
                'median_fitness': median_fitness,
                'p5_fitness': p5_fitness,
                'p95_fitness': p95_fitness
            })
            
            # Create progress plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            generations = [d['generation'] for d in self.training_progress_data]
            archive_sizes = [d['archive_size'] for d in self.training_progress_data]
            min_fitnesses = [d['min_fitness'] for d in self.training_progress_data]
            mean_fitnesses = [d['mean_fitness'] for d in self.training_progress_data]
            median_fitnesses = [d['median_fitness'] for d in self.training_progress_data]
            prompt_sizes = [d['prompt_archive_size'] for d in self.training_progress_data]
            
            # Archive size over generations
            axes[0, 0].plot(generations, archive_sizes, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('Generation')
            axes[0, 0].set_ylabel('Archive Size')
            axes[0, 0].set_title('Archive Growth')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Fitness evolution
            axes[0, 1].plot(generations, min_fitnesses, 'r-o', label='Best', linewidth=2, markersize=6)
            axes[0, 1].plot(generations, mean_fitnesses, 'g-o', label='Mean', linewidth=2, markersize=6)
            axes[0, 1].plot(generations, median_fitnesses, 'b-o', label='Median', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Fitness (Validation Loss)')
            axes[0, 1].set_title('Fitness Evolution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Prompt archive size
            axes[1, 0].plot(generations, prompt_sizes, 'orange', linewidth=2, marker='o', markersize=6)
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Prompt Archive Size')
            axes[1, 0].set_title('Prompt Archive Growth')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Archive coverage (if we have total niches info)
            if hasattr(self, 'total_niches') and self.total_niches:
                coverage = [size / self.total_niches * 100 for size in archive_sizes]
                axes[1, 1].plot(generations, coverage, 'purple', linewidth=2, marker='o', markersize=6)
                axes[1, 1].set_xlabel('Generation')
                axes[1, 1].set_ylabel('Archive Coverage (%)')
                axes[1, 1].set_title('Archive Coverage')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'live_training_progress.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not update training progress plot: {e}")
    
    def update_pipeline_effectiveness_plot(self, generation, total_candidates, valid_candidates, 
                                         added_to_archive, success_rate, archive_coverage):
        """
        Update and save the pipeline effectiveness plot in real-time.
        """
        try:
            # Add new data
            self.pipeline_effectiveness_data.append({
                'generation': generation,
                'total_candidates': total_candidates,
                'valid_candidates': valid_candidates,
                'added_to_archive': added_to_archive,
                'success_rate': success_rate,
                'archive_coverage': archive_coverage
            })
            
            # Create effectiveness plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            generations = [d['generation'] for d in self.pipeline_effectiveness_data]
            success_rates = [d['success_rate'] for d in self.pipeline_effectiveness_data]
            archive_coverages = [d['archive_coverage'] for d in self.pipeline_effectiveness_data]
            total_candidates_list = [d['total_candidates'] for d in self.pipeline_effectiveness_data]
            valid_candidates_list = [d['valid_candidates'] for d in self.pipeline_effectiveness_data]
            added_list = [d['added_to_archive'] for d in self.pipeline_effectiveness_data]
            
            # Success rate over generations
            axes[0, 0].plot(generations, success_rates, 'b-o', linewidth=2, markersize=6)
            axes[0, 0].set_xlabel('Generation')
            axes[0, 0].set_ylabel('Success Rate (%)')
            axes[0, 0].set_title('Pipeline Success Rate')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_ylim(0, 100)
            
            # Archive coverage over generations
            axes[0, 1].plot(generations, archive_coverages, 'g-o', linewidth=2, markersize=6)
            axes[0, 1].set_xlabel('Generation')
            axes[0, 1].set_ylabel('Archive Coverage (%)')
            axes[0, 1].set_title('Archive Coverage')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 100)
            
            # Candidate counts
            x_pos = np.arange(len(generations))
            width = 0.25
            axes[1, 0].bar(x_pos - width, total_candidates_list, width, alpha=0.7, label='Total')
            axes[1, 0].bar(x_pos, valid_candidates_list, width, alpha=0.7, label='Valid')
            axes[1, 0].bar(x_pos + width, added_list, width, alpha=0.7, label='Added to Archive')
            axes[1, 0].set_xlabel('Generation')
            axes[1, 0].set_ylabel('Number of Networks')
            axes[1, 0].set_title('Network Generation Statistics')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels(generations)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Efficiency metrics
            efficiency = [(added / total * 100) if total > 0 else 0 
                         for added, total in zip(added_list, total_candidates_list)]
            axes[1, 1].plot(generations, efficiency, 'r-o', linewidth=2, markersize=6)
            axes[1, 1].set_xlabel('Generation')
            axes[1, 1].set_ylabel('Efficiency (%)')
            axes[1, 1].set_title('Archive Addition Efficiency')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_ylim(0, 100)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'live_pipeline_effectiveness.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not update pipeline effectiveness plot: {e}")
    

def create_plots(state: dict, save_path: str, cfg: Config):
 
    print("###Creating Plots###")
    
    if not state['nets_archive']:
        print("No networks in archive to plot")
        return
    
    # Extract data
    diversity_scores = []
    efficiency_ratios = []
    fitnesses = []
    network_types = []
    
    for species in state['nets_archive'].values():
        fitnesses.append(species.fitness)
        network_types.append('pregen' if 'pregen_network' in str(species.net_path) else 'generated')
        
        if len(species.desc) >= 2:
            diversity_scores.append(species.desc[0])
            efficiency_ratios.append(species.desc[1])
        else:
            # Fallback to default values if descriptors are missing
            diversity_scores.append(0.5)
            efficiency_ratios.append(0.5)
    
    # Create plots directory
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Descriptor Distribution Plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Diversity vs Efficiency scatter
    scatter = axes[0, 0].scatter(diversity_scores, efficiency_ratios, c=fitnesses, 
                                cmap='viridis', s=100, alpha=0.8, edgecolors='black')
    axes[0, 0].set_xlabel('Architectural Diversity Score')
    axes[0, 0].set_ylabel('Efficiency Ratio')
    axes[0, 0].set_title('Descriptor Space Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[0, 0])
    cbar.set_label('Fitness (Validation Loss)')
    
    # Fitness distribution
    axes[0, 1].hist(fitnesses, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_xlabel('Fitness (Validation Loss)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Fitness Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Diversity distribution
    axes[1, 0].hist(diversity_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Architectural Diversity Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Diversity Score Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency distribution
    axes[1, 1].hist(efficiency_ratios, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[1, 1].set_xlabel('Efficiency Ratio')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Efficiency Ratio Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'descriptor_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Archive Coverage Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Use the config value for total niches 
    total_niches = cfg.N_NICHES  
    
    # Calculate optimal grid dimensions
    # e.g For 25 niches, we want a 5x5 grid
    grid_cols = int(np.ceil(np.sqrt(total_niches)))
    grid_rows = int(np.ceil(total_niches / grid_cols))
    
    print(f"Creating {grid_rows}x{grid_cols} grid for {total_niches} niches")
    
    # Create a 2D grid to show niche coverage
    niche_coverage = np.zeros((grid_rows, grid_cols))
    
    # Extract descriptor data for plotting
    diversity_scores = []
    efficiency_ratios = []
    
    for species in state['nets_archive'].values():
        if len(species.desc) >= 2:
            # Use the first two descriptor values for plotting
            diversity_scores.append(species.desc[0])
            efficiency_ratios.append(species.desc[1])
        else:
            # Fallback to default values
            diversity_scores.append(0.5)
            efficiency_ratios.append(0.5)
    
    if not diversity_scores or not efficiency_ratios:
        print("No valid descriptor data found for plotting")
        return
    
    # Normalize descriptors to 0-1 range for grid mapping
    div_min, div_max = min(diversity_scores), max(diversity_scores)
    eff_min, eff_max = min(efficiency_ratios), max(efficiency_ratios)
    
    # Avoid division by zero
    div_range = div_max - div_min if div_max != div_min else 1.0
    eff_range = eff_max - eff_min if eff_max != eff_min else 1.0
    
    # Map networks to grid positions based on normalized descriptors
    mapped_count = 0
    used_positions = set()  # Track which grid positions are already used
    
    # Sort networks by fitness to prioritize better networks for better positions
    sorted_networks = sorted(state['nets_archive'].values(), key=lambda x: x.fitness)
    
    for species in sorted_networks:
        try:
            if len(species.desc) >= 2:
                # Normalize descriptors
                div_norm = (species.desc[0] - div_min) / div_range
                eff_norm = (species.desc[1] - eff_min) / eff_range
                
                # Try to find the best available position
                best_position = None
                best_distance = float('inf')
                
                # First, try the ideal position based on descriptors
                ideal_x = int(div_norm * (grid_cols - 1))
                ideal_y = int(eff_norm * (grid_rows - 1))
                ideal_x = max(0, min(ideal_x, grid_cols - 1))
                ideal_y = max(0, min(ideal_y, grid_rows - 1))
                
                # Check if ideal position is available
                if (ideal_y, ideal_x) not in used_positions:
                    best_position = (ideal_y, ideal_x)
                else:
                    # Find the closest available position
                    for y in range(grid_rows):
                        for x in range(grid_cols):
                            if (y, x) not in used_positions:
                                # Calculate distance from ideal position
                                distance = ((y - ideal_y) ** 2 + (x - ideal_x) ** 2) ** 0.5
                                if distance < best_distance:
                                    best_distance = distance
                                    best_position = (y, x)
                
                if best_position is not None:
                    grid_y, grid_x = best_position
                    used_positions.add(best_position)
                    niche_coverage[grid_y, grid_x] = species.fitness
                    mapped_count += 1
                else:
                    print(f"Warning: No available position for network")
                
            else:
                print(f"Warning: Network has insufficient descriptor data: {len(species.desc)} dimensions")
        except Exception as e:
            print(f"Warning: Could not map network to grid position: {e}")
            continue
    
    print(f"Successfully mapped {mapped_count} networks to grid positions")
    print(f"Used {len(used_positions)} unique grid positions")
    
    # Verify we have the right number of networks mapped
    expected_count = len(state['nets_archive'])
    if mapped_count != expected_count:
        print(f"Warning: Expected {expected_count} networks but mapped {mapped_count}")
    
    im = ax.imshow(niche_coverage, cmap='viridis', aspect='auto')
    ax.set_xlabel('Diversity Score (Normalized)')
    ax.set_ylabel('Efficiency Ratio (Normalized)')
    ax.set_title(f'Archive Coverage Map ({len(state["nets_archive"])}/{total_niches} niches filled)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fitness (Validation Loss)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'archive_coverage_map.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Network Type Analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Network type distribution
    type_counts = {'pregen': network_types.count('pregen'), 'generated': network_types.count('generated')}
    axes[0].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Network Type Distribution')
    
    # Fitness by network type
    pregen_fitnesses = [f for f, t in zip(fitnesses, network_types) if t == 'pregen']
    generated_fitnesses = [f for f, t in zip(fitnesses, network_types) if t == 'generated']
    
    if pregen_fitnesses and generated_fitnesses:
        axes[1].boxplot([pregen_fitnesses, generated_fitnesses], labels=['Pregen', 'Generated'])
        axes[1].set_ylabel('Fitness (Validation Loss)')
        axes[1].set_title('Fitness by Network Type')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'network_type_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Statistics Summary
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.axis('off')
    
    stats_text = f"""
    Archive Statistics Summary
    
    Total Networks: {len(state['nets_archive'])}
    Archive Coverage: {len(state['nets_archive'])}/{total_niches} ({len(state['nets_archive'])/total_niches*100:.1f}%)
    
    Fitness Statistics:
    - Best: {min(fitnesses):.4f}
    - Worst: {max(fitnesses):.4f}
    - Mean: {np.mean(fitnesses):.4f}
    - Std: {np.std(fitnesses):.4f}
    
    Descriptor Statistics:
    - Diversity Range: [{min(diversity_scores):.4f}, {max(diversity_scores):.4f}]
    - Efficiency Range: [{min(efficiency_ratios):.4f}, {max(efficiency_ratios):.4f}]
    
    Network Types:
    - Pregen: {network_types.count('pregen')}
    - Generated: {network_types.count('generated')}
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=12, 
            verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'statistics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Plots saved to: {plots_dir}")
    print(f"  - Descriptor distributions")
    print(f"  - Archive coverage map")
    print(f"  - Network type analysis")
    print(f"  - Statistics summary")