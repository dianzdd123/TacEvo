import json
import matplotlib.pyplot as plt
from pathlib import Path

def load_trainability_data(path):
    data = {}
    files = list(Path(path).glob("all_networks_gen_*.json"))
    
    for file_path in files:
        try:
            filename = file_path.name
            gen_str = filename.replace("all_networks_gen_", "").replace(".json", "")
            generation = int(gen_str)
            
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            total_networks = json_data.get("total_networks")
            if total_networks is not None:
                data[generation] = total_networks
        except (json.JSONDecodeError, ValueError, KeyError):
            continue
    
    return data

def plot_trainability():
    force_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\force_20_gens_50_nets\force_20_gens_50_nets"
    grating_path = r"D:\Masters_Dissertation\tactile_nas_deployment\logs\experiments\grating_20_gens_50_nets\grating_20_gens_50_nets"
    
    force_data = load_trainability_data(force_path)
    grating_data = load_trainability_data(grating_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    force_gens = sorted(force_data.keys())
    grating_gens = sorted(grating_data.keys())
    
    if force_gens:
        force_networks = [force_data[gen] for gen in force_gens]
        ax.plot([gen + 1 for gen in force_gens], force_networks, 'o-', color='blue', linewidth=2, markersize=6, label='Force')
    
    if grating_gens:
        grating_networks = [grating_data[gen] for gen in grating_gens]
        ax.plot([gen + 1 for gen in grating_gens], grating_networks, 's-', color='orange', linewidth=2, markersize=6, label='Grating')
    
    ax.set_title('Trainable Networks vs Generation', fontsize=20, fontweight='bold')
    ax.set_xlabel('Generation', fontsize=16)
    ax.set_ylabel('Number of Trainable Networks', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Max Generated Networks')
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(0, 55)
    ax.set_xticks(range(1, max(max(force_gens) if force_gens else 0, max(grating_gens) if grating_gens else 0) + 2))
    
    if force_gens:
        force_avg = sum(force_data.values()) / len(force_data.values())
        force_success_rate = (force_avg / 50) * 100
    else:
        force_success_rate = 0
        
    if grating_gens:
        grating_avg = sum(grating_data.values()) / len(grating_data.values())
        grating_success_rate = (grating_avg / 50) * 100
    else:
        grating_success_rate = 0
    
    textstr = f'Average Success Rate\nForce: {force_success_rate:.1f}%\nGrating: {grating_success_rate:.1f}%'
    props = dict(boxstyle='round', facecolor='orange', alpha=0.7)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=props, color='black', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("trainable_networks.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_trainability()
