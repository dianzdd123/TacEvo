import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(path, data_type):
    data = {}
    files = list(Path(path).glob(f"{data_type}_gen_*.json"))
    
    for file_path in files:
        filename = file_path.name
        gen_str = filename.replace(f"{data_type}_gen_", "").replace(".json", "")
        generation = int(gen_str)
        
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        if data_type == "archive_map":
            size = json_data.get("archive_size")
        else:
            size = json_data.get("prompt_archive_size")
            
        if size is not None:
            data[generation] = size
    
    return data

def plot_growth():
    force_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\force_20_gens_50_nets\force_20_gens_50_nets"
    grating_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\grating_20_gens_50_nets\grating_20_gens_50_nets"
    
    force_archive = load_data(force_path, 'archive_map')
    force_prompt = load_data(force_path, 'prompt_archive')
    grating_archive = load_data(grating_path, 'archive_map')
    grating_prompt = load_data(grating_path, 'prompt_archive')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    force_gens = sorted(force_archive.keys())
    grating_gens = sorted(grating_archive.keys())
    
    if force_gens:
        force_sizes = [force_archive[gen] for gen in force_gens]
        ax1.plot([gen + 1 for gen in force_gens], force_sizes, 'o-', color='blue', linewidth=2, markersize=6, label='Force')
    
    if grating_gens:
        grating_sizes = [grating_archive[gen] for gen in grating_gens]
        ax1.plot([gen + 1 for gen in grating_gens], grating_sizes, 's-', color='orange', linewidth=2, markersize=6, label='Grating')
    
    ax1.set_title('Network Archive Growth', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Archive Size', fontsize=12)
    ax1.set_ylim(0, 25)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_xticks(range(1, max(max(force_gens) if force_gens else 0, max(grating_gens) if grating_gens else 0) + 2))
    
    # Add coverage percentage box for network archive
    if force_gens or grating_gens:
        max_gen = max(max(force_gens) if force_gens else 0, max(grating_gens) if grating_gens else 0)
        force_coverage = (force_archive.get(max_gen, 0) / 25) * 100 if force_gens else 0
        grating_coverage = (grating_archive.get(max_gen, 0) / 25) * 100 if grating_gens else 0
        
        coverage_text = f'Coverage %:\nForce: {force_coverage:.1f}%\nGrating: {grating_coverage:.1f}%'
        ax1.text(0.95, 0.05, coverage_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                fontsize=14, ha='right', va='bottom')
    
    force_prompt_gens = sorted(force_prompt.keys())
    grating_prompt_gens = sorted(grating_prompt.keys())
    
    if force_prompt_gens:
        force_prompt_sizes = [force_prompt[gen] for gen in force_prompt_gens]
        ax2.plot([gen + 1 for gen in force_prompt_gens], force_prompt_sizes, 'o-', color='blue', linewidth=2, markersize=6, label='Force')
    
    if grating_prompt_gens:
        grating_prompt_sizes = [grating_prompt[gen] for gen in grating_prompt_gens]
        ax2.plot([gen + 1 for gen in grating_prompt_gens], grating_prompt_sizes, 's-', color='orange', linewidth=2, markersize=6, label='Grating')
    
    ax2.set_title('Prompt Archive Growth', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Prompt Archive Size', fontsize=12)
    ax2.set_ylim(0, 25)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.set_xticks(range(1, max(max(force_prompt_gens) if force_prompt_gens else 0, max(grating_prompt_gens) if grating_prompt_gens else 0) + 2))
    
    # Add coverage percentage box for prompt archive
    if force_prompt_gens or grating_prompt_gens:
        max_gen = max(max(force_prompt_gens) if force_prompt_gens else 0, max(grating_prompt_gens) if grating_prompt_gens else 0)
        force_prompt_coverage = (force_prompt.get(max_gen, 0) / 25) * 100 if force_prompt_gens else 0
        grating_prompt_coverage = (grating_prompt.get(max_gen, 0) / 25) * 100 if grating_prompt_gens else 0
        
        coverage_text = f'Coverage %:\nForce: {force_prompt_coverage:.1f}%\nGrating: {grating_prompt_coverage:.1f}%'
        ax2.text(0.95, 0.05, coverage_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
                fontsize=14, ha='right', va='bottom')
    
    plt.tight_layout()
    plt.savefig("archive_growth_comparison.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_growth()