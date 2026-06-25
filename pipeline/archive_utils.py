import os
import json
import csv
from datetime import datetime
import numpy as np
from conf.config import Config
from map_elites import common as cm

#  Archive Helper Functions 
def __add_to_archive(s, centroid, archive, kdt, type_ind):

    
    # Query the KDTree to find the nearest centroid
    distances, indices = kdt.query([centroid], k=1)
    niche_index = indices[0][0]
    distance_to_niche = distances[0][0]
    
    # Get the actual niche centroid from the KDTree data
    niche = kdt.data[niche_index]
  
    # Calculate distances to all niches for debugging
    all_distances, all_indices = kdt.query([centroid], k=len(kdt.data))
    
    # Find the 3 closest niches
    closest_3_indices = all_indices[0][:3]
    closest_3_distances = all_distances[0][:3]
    
    # Make the niche hashable
    n = cm.make_hashable(niche)
    s.centroid = n
    
    if n in archive:
        print(f"      Existing fitness: {archive[n].fitness}, New fitness: {s.fitness}")
        if type_ind == "network" and s.fitness < archive[n].fitness:
            print(f"Replacing network (better fitness)")
            archive[n] = s
            return True
        elif type_ind == "prompt" and s.fitness > archive[n].fitness:
            print(f"Replacing prompt (better fitness)")
            archive[n] = s
            return True
        print(f"Not replacing (worse fitness)")
        return False
    else:
        print(f"Adding new network to archive")
        archive[n] = s
        return True

def to_specie(net, fit, desc, net_path):
    return cm.Species(net, desc, fit, net_path)

#  Custom Archive Saving Function 
def save_archive_to_path(archive, gen, name, save_path):
    """
    Custom function to save archive files to a specific path.
    """
    def write_array(a, f):
        if a is not None:
            for i in a:
                f.write(str(i) + " ")
        else:
            f.write("0.0 0.0 ")  # Default values if centroid is None
    
    # Normalize the path to fix Windows path issues
    normalized_save_path = os.path.normpath(save_path)
    filename = os.path.join(normalized_save_path, f"{name}archive_{gen}.dat")
    
    # Ensure the directory exists
    os.makedirs(normalized_save_path, exist_ok=True)
    
    with open(filename, 'w') as f:
        for k in archive.values():
            f.write(k.net_path + " ")
            f.write(str(k.fitness) + ' ')
            write_array(k.centroid, f)
            f.write(str(k.curiosity) + " ")
            f.write(str(k.desc[0]) + " ")
            f.write(str(k.desc[1]) + " ")
            f.write("\n")

def save_archive_results(state: dict, cfg: Config, save_path: str):
    """
    Save archive results including all networks, prompts, descriptors, and fitness scores.
    """
    print(" Saving Archive Results ")
    
    # Create archive results directory
    archive_results_dir = os.path.join(save_path, "archive_results")
    os.makedirs(archive_results_dir, exist_ok=True)
    
    # Save archive networks data
    archive_networks = []
    for niche_key, species in state['nets_archive'].items():
        network_data = {
            'niche_key': str(niche_key),
            'fitness': species.fitness,
            'descriptors': {
                'diversity_score': species.desc[0],
                'efficiency_ratio': species.desc[1]
            },
            'network_path': species.net_path if hasattr(species, 'net_path') else 'pregen_network',
            'network_type': 'pregen' if 'pregen_network' in str(species.net_path) else 'generated',
            'parameter_count': sum(p.numel() for p in species.net.parameters()) if hasattr(species, 'net') else 0,
            'timestamp': datetime.now().isoformat()
        }
        archive_networks.append(network_data)
    
    # Save networks data
    networks_file = os.path.join(archive_results_dir, "archive_networks.json")
    with open(networks_file, 'w') as f:
        json.dump(archive_networks, f, indent=2, default=str)
    
    # Save prompt archive data
    prompt_archive = []
    for prompt_key, prompt_species in state['prompt_archive'].items():
        prompt_data = {
            'prompt_key': str(prompt_key),
            'curiosity_score': prompt_species.curiosity,
            'descriptors': {
                'diversity_score': prompt_species.desc[0],
                'efficiency_ratio': prompt_species.desc[1]
            },
            'prompt_text': state['int_to_prompt'][int(prompt_species.desc[0] * (len(state['prompts']) - 1))] if 'int_to_prompt' in state else 'unknown',
            'timestamp': datetime.now().isoformat()
        }
        prompt_archive.append(prompt_data)
    
    # Save prompt archive data
    prompts_file = os.path.join(archive_results_dir, "prompt_archive.json")
    with open(prompts_file, 'w') as f:
        json.dump(prompt_archive, f, indent=2, default=str)
    
    # Save experiment summary
    summary_data = {
        'experiment_config': {
            'mutation_model': cfg.MUTATION,
            'num_niches': cfg.N_NICHES,
            'training_epochs': cfg.NET_TRAINING_EPOCHS,
            'temperature': cfg.TEMPERATURE,
            'num_networks_per_generation': cfg.NUM_NETS
        },
        'archive_statistics': {
            'total_networks': len(state['nets_archive']),
            'total_prompts': len(state['prompt_archive']),
            'archive_coverage': f"{len(state['nets_archive'])}/{cfg.N_NICHES} ({len(state['nets_archive'])/cfg.N_NICHES*100:.1f}%)",
            'best_fitness': min([s.fitness for s in state['nets_archive'].values()]) if state['nets_archive'] else float('inf'),
            'worst_fitness': max([s.fitness for s in state['nets_archive'].values()]) if state['nets_archive'] else 0,
            'mean_fitness': np.mean([s.fitness for s in state['nets_archive'].values()]) if state['nets_archive'] else 0,
            'fitness_std': np.std([s.fitness for s in state['nets_archive'].values()]) if state['nets_archive'] else 0
        },
        'descriptor_statistics': {
            'diversity_scores': [s.desc[0] for s in state['nets_archive'].values()],
            'efficiency_ratios': [s.desc[1] for s in state['nets_archive'].values()],
            'diversity_range': [min([s.desc[0] for s in state['nets_archive'].values()]), max([s.desc[0] for s in state['nets_archive'].values()])] if state['nets_archive'] else [0, 0],
            'efficiency_range': [min([s.desc[1] for s in state['nets_archive'].values()]), max([s.desc[1] for s in state['nets_archive'].values()])] if state['nets_archive'] else [0, 0]
        },
        'experiment_timestamp': datetime.now().isoformat()
    }
    
    # Save summary
    summary_file = os.path.join(archive_results_dir, "experiment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Save CSV for easy analysis
    csv_file = os.path.join(archive_results_dir, "archive_networks.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Network_ID', 'Fitness', 'Diversity_Score', 'Efficiency_Ratio', 'Parameter_Count', 'Network_Type', 'Niche_Key'])
        for i, network in enumerate(archive_networks):
            writer.writerow([
                i+1,
                network['fitness'],
                network['descriptors']['diversity_score'],
                network['descriptors']['efficiency_ratio'],
                network['parameter_count'],
                network['network_type'],
                network['niche_key']
            ])
    
    print(f"Archive results saved to: {archive_results_dir}")
    print(f"- Networks: {len(archive_networks)} networks")
    print(f"- Prompts: {len(prompt_archive)} prompts")
    print(f"- Coverage: {summary_data['archive_statistics']['archive_coverage']}")
    print(f"- Best fitness: {summary_data['archive_statistics']['best_fitness']:.4f}")
    
    return archive_results_dir