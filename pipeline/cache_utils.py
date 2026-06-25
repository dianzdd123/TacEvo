import os
import pickle
import hashlib
import json
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from conf.config import Config


class ArchiveCacheManager:
    """
    Manages caching of archive initialization results to avoid re-running expensive
    init_archive_pregen processes for the same configuration.
    """
    
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cache_dir = self._get_cache_directory()
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_directory(self) -> str:
        """Get the cache directory path."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "cache", "archive_cache")
    
    def _generate_cache_key(self) -> str:
        """
        Generate a unique cache key based on configuration parameters.
        This ensures cache is specific to the exact configuration being used.
        """
        # Create a hash of the relevant configuration parameters
        config_hash_data = {
            'network_type': self.cfg.NETWORK_TYPE,
            'task_type': self.cfg.TASK_TYPE,
            'n_niches': self.cfg.N_NICHES,
            'dim_map': self.cfg.DIM_MAP,
            'net_training_epochs': self.cfg.NET_TRAINING_EPOCHS,
            'learning_rate': self.cfg.LEARNING_RATE,
            'batch_size': self.cfg.BATCH_SIZE,
            'momentum': self.cfg.MOMENTUM,
            'num_classes': self.cfg.NUM_CLASSES,
            'output_size': self.cfg.OUTPUT_SIZE,
            'img_channels': self.cfg.IMG_CHANNELS,
            'num_layers': self.cfg.NUM_LAYERS,
            'bounds_padding_percent': self.cfg.BOUNDS_PADDING_PERCENT,
            'min_padding': self.cfg.MIN_PADDING
        }
        
        # Convert to JSON string and hash it
        config_str = json.dumps(config_hash_data, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        # Include task type in the filename for easy identification
        cache_key = f"{self.cfg.TASK_TYPE}_{config_hash}"
        return cache_key
    
    def _get_cache_file_path(self, cache_key: str) -> str:
        """Get the full path for the cache file."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _get_cache_metadata_path(self, cache_key: str) -> str:
        """Get the full path for the cache metadata file."""
        return os.path.join(self.cache_dir, f"{cache_key}_metadata.json")
    
    def cache_exists(self) -> bool:
        """
        Check if a valid cache exists for the current configuration.
        Returns True if cache exists and is valid, False otherwise.
        """
        # Check if caching is disabled
        if not self.cfg.USE_ARCHIVE_CACHE:
            return False
            
        # Check if force refresh is enabled
        if self.cfg.FORCE_CACHE_REFRESH:
            print("Force cache refresh enabled, will regenerate cache")
            return False
            
        cache_key = self._generate_cache_key()
        cache_file = self._get_cache_file_path(cache_key)
        metadata_file = self._get_cache_metadata_path(cache_key)
        
        # Check if both files exist
        if not (os.path.exists(cache_file) and os.path.exists(metadata_file)):
            return False
        
        # Check if cache is not too old
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                cache_age_days = (datetime.now() - datetime.fromisoformat(metadata['created_at'])).days
                if cache_age_days > self.cfg.CACHE_EXPIRY_DAYS:
                    print(f"Cache is {cache_age_days} days old (expires after {self.cfg.CACHE_EXPIRY_DAYS} days), will regenerate")
                    return False
        except Exception as e:
            print(f"Error reading cache metadata: {e}")
            return False
        
        return True
    
    def load_cache(self) -> Optional[Dict[str, Any]]:
        """
        Load the cached archive state.
        Returns the cached state dictionary or None if loading fails.
        """
        cache_key = self._generate_cache_key()
        cache_file = self._get_cache_file_path(cache_key)
        metadata_file = self._get_cache_metadata_path(cache_key)
        
        try:
            # Load metadata first
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            print(f"Loading cache for {self.cfg.TASK_TYPE} task (created: {metadata['created_at']})")
            
            # Load the cached state
            with open(cache_file, 'rb') as f:
                cached_state = pickle.load(f)
            
            print(f"Successfully loaded cache with {len(cached_state.get('nets_archive', {}))} networks")
            return cached_state
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def save_cache(self, state: Dict[str, Any]) -> bool:
        """
        Save the archive state to cache.
        Returns True if successful, False otherwise.
        """
        cache_key = self._generate_cache_key()
        cache_file = self._get_cache_file_path(cache_key)
        metadata_file = self._get_cache_metadata_path(cache_key)
        
        try:
            # Create a clean copy of the state for caching
            # Remove non-picklable objects
            cache_state = {}
            for key, value in state.items():
                if key in ['log_file', 'logger', 'live_plotter', 'curios_net', 'net_class']:
                    # Skip non-picklable objects
                    continue
                elif key == 'nets_archive':
                    # Deep copy the nets_archive but remove network instances
                    cache_state[key] = {}
                    for niche_key, species in value.items():
                        # Create a copy of the species without the network instance
                        species_copy = type(species)(
                            x=None,  # Remove the network instance (x attribute)
                            desc=species.desc.copy() if hasattr(species.desc, 'copy') else species.desc,
                            fitness=species.fitness,
                            net_path=species.net_path,
                            centroid=getattr(species, 'centroid', None)
                        )
                        # Set curiosity separately as it's not in __init__
                        species_copy.curiosity = getattr(species, 'curiosity', 0.0)
                        cache_state[key][niche_key] = species_copy
                elif key == 'prompt_archive':
                    # Deep copy the prompt_archive
                    cache_state[key] = {}
                    for prompt_key, species in value.items():
                        species_copy = type(species)(
                            x=None,
                            desc=species.desc.copy() if hasattr(species.desc, 'copy') else species.desc,
                            fitness=species.fitness,
                            net_path=species.net_path,
                            centroid=getattr(species, 'centroid', None)
                        )
                        # Set curiosity separately as it's not in __init__
                        species_copy.curiosity = getattr(species, 'curiosity', 0.0)
                        cache_state[key][prompt_key] = species_copy
                else:
                    # Copy other values directly
                    cache_state[key] = value
            
            # Create metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'task_type': self.cfg.TASK_TYPE,
                'network_type': self.cfg.NETWORK_TYPE,
                'n_niches': self.cfg.N_NICHES,
                'archive_size': len(cache_state.get('nets_archive', {})),
                'config_hash': cache_key
            }
            
            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save the clean state
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_state, f)
            
            print(f"Successfully cached archive state for {self.cfg.TASK_TYPE} task")
            print(f"Cache saved to: {cache_file}")
            return True
            
        except Exception as e:
            print(f"Error saving cache: {e}")
            return False
    
    def clear_cache(self) -> bool:
        """
        Clear all cache files for the current configuration.
        Returns True if successful, False otherwise.
        """
        cache_key = self._generate_cache_key()
        cache_file = self._get_cache_file_path(cache_key)
        metadata_file = self._get_cache_metadata_path(cache_key)
        
        try:
            if os.path.exists(cache_file):
                os.remove(cache_file)
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            print(f"Cleared cache for {self.cfg.TASK_TYPE} task")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache status.
        Returns a dictionary with cache information.
        """
        cache_key = self._generate_cache_key()
        cache_file = self._get_cache_file_path(cache_key)
        metadata_file = self._get_cache_metadata_path(cache_key)
        
        info = {
            'cache_key': cache_key,
            'cache_file': cache_file,
            'metadata_file': metadata_file,
            'exists': os.path.exists(cache_file),
            'task_type': self.cfg.TASK_TYPE,
            'network_type': self.cfg.NETWORK_TYPE
        }
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                info.update(metadata)
            except Exception as e:
                info['metadata_error'] = str(e)
        
        return info


def test_cache_system():
    """
    Test function to verify the caching system works correctly.
    """
    from conf.config import Config
    
    print("=== Testing Archive Cache System ===")
    
    # Create test configurations
    test_configs = [
        Config(TASK_TYPE="shape", NETWORK_TYPE="single_label"),
        Config(TASK_TYPE="CIFAR-10", NETWORK_TYPE="resnet"),
        Config(TASK_TYPE="force", NETWORK_TYPE="image_regression"),
        Config(TASK_TYPE="pose", NETWORK_TYPE="image_regression")
    ]
    
    for i, cfg in enumerate(test_configs):
        print(f"\n Test {i+1}: {cfg.TASK_TYPE} task with {cfg.NETWORK_TYPE} ")
        
        # Create cache manager
        cache_manager = ArchiveCacheManager(cfg)
        
        # Get cache info
        cache_info = cache_manager.get_cache_info()
        print(f"Cache key: {cache_info['cache_key']}")
        print(f"Cache exists: {cache_info['exists']}")
        
        # Test cache operations
        if cache_info['exists']:
            print("Loading existing cache")
            cached_state = cache_manager.load_cache()
            if cached_state:
                print(f"Successfully loaded cache with {len(cached_state.get('nets_archive', {}))} networks")
            else:
                print("Failed to load cache")
        
        # Test saving a dummy cache
        print("Testing cache save")
        dummy_state = {
            'nets_archive': {'dummy': 'data'},
            'descriptor_bounds': {'test': 'bounds'},
            'kdt': None
        }
        
        if cache_manager.save_cache(dummy_state):
            print("Successfully saved test cache")
            
            # Test loading the saved cache
            loaded_state = cache_manager.load_cache()
            if loaded_state:
                print("Successfully loaded saved cache")
            else:
                print("Failed to load saved cache")
        else:
            print("Failed to save test cache")
    
    print("\n=== Cache System Test Complete ===")


if __name__ == "__main__":
    test_cache_system() 