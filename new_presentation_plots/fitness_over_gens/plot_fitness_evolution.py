import numpy as np
import glob
import json
import os
import matplotlib.pyplot as plt

def process_experiment_data(experiment_path):
    pattern = os.path.join(experiment_path, "archive_map_gen_*.json")
    archive_files = glob.glob(pattern)
    archive_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))

    generations, bests, medians = [], [], []

    for file_path in archive_files:
        gen_num = int(os.path.basename(file_path).split('_')[-1].split('.')[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        fitness_values = [net['fitness'] for net in data.get('networks', []) if net.get('fitness') is not None]
        
        if fitness_values:
            generations.append(gen_num)
            bests.append(np.min(fitness_values))
            medians.append(np.median(fitness_values))
            
    return np.array(generations), np.array(bests), np.array(medians)

def plot_combined_fitness_evolution(force_generations, force_bests, force_medians, 
                                   grating_generations, grating_bests, grating_medians):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Force experiment subplot
    force_display_generations = force_generations + 1
    
    # Plot median fitness
    ax1.plot(force_display_generations, force_medians, color='red', label='Median Fitness', linewidth=2)
    
    # Plot best fitness
    ax1.plot(force_display_generations, force_bests, color='firebrick', label='Best Fitness', linewidth=2.5)
    
    ax1.set_xlabel('Generation', fontsize=14)
    ax1.set_ylabel('Fitness (MSE Validation Loss)', fontsize=14)
    ax1.set_title('Force Experiment - Network Archive Fitness Evolution', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right', frameon=True)
    ax1.set_xlim(min(force_display_generations), max(force_display_generations)+1)
    ax1.set_ylim(bottom=0)
    
    force_improvement = ((force_bests[0] - force_bests.min()) / force_bests[0]) * 100
    textstr = f'Relative improvement: {force_improvement:.1f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

    # Grating experiment subplot
    grating_display_generations = grating_generations + 1
    
    # Plot median fitness
    ax2.plot(grating_display_generations, grating_medians, color='blue', label='Median Fitness', linewidth=2)
    
    # Plot best fitness
    ax2.plot(grating_display_generations, grating_bests, color='darkblue', label='Best Fitness', linewidth=2.5)
    
    ax2.set_xlabel('Generation', fontsize=14)
    ax2.set_ylabel('Fitness (Cross-Entropy Validation Loss)', fontsize=14)
    ax2.set_title('Grating Experiment - Network Archive Fitness Evolution', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper right', frameon=True)
    ax2.set_xlim(min(grating_display_generations), max(grating_display_generations)+1)
    ax2.set_ylim(bottom=0)
    
    grating_improvement = ((grating_bests[0] - grating_bests.min()) / grating_bests[0]) * 100
    textstr = f'Relative improvement: {grating_improvement:.1f}%'
    props = dict(boxstyle='round,pad=0.5', facecolor='orange', alpha=0.8)
    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    plt.subplots_adjust(hspace=0.4)

    plt.savefig('combined_fitness_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Force experiment path
    force_experiment_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\force_20_gens_50_nets\force_20_gens_50_nets"
    
    # Grating experiment path
    grating_experiment_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\grating_20_gens_50_nets\grating_20_gens_50_nets"
    
    # Process both datasets
    force_generations, force_bests, force_medians = process_experiment_data(force_experiment_path)
    grating_generations, grating_bests, grating_medians = process_experiment_data(grating_experiment_path)
    
    if force_generations.size > 0 and grating_generations.size > 0:
        plot_combined_fitness_evolution(force_generations, force_bests, force_medians,
                                       grating_generations, grating_bests, grating_medians)
    else:
        print("Error: Could not load data from one or both experiments")