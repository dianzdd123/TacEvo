from dataclasses import dataclass
import typing
import multiprocessing as mp

from hydra.core.config_store import ConfigStore
from map_elites import common as cm

@dataclass
class Config:
    """This class provides a "schema" for the config file, validating types."""

    RANDOM_INIT_NETS: int = 50 # Not Being Used After Pregen Implementation
    INIT_NUM_NETS: int = 20 # Not Being Used After Pregen Implementation
    ROLL_OUTS: int = 2 # NOT USED ANYMORE
    NUM_NETS: int = 50 # Mutation and crossover nets to be created (increased for better evolution)
    START_FROM_CHECKPOINT: bool = False
    RANDOM_NETWORKS: bool = False
    # Base Network Type
    NETWORK_TYPE: str = "image_regression" # "resnet" or "image_regression" or "single_label"

    # Task Type
    TASK_TYPE: str = "force" # "shape" or "CIFAR-10" or "grating" or "force" or "pose" or "shape_texture_stiffness_material"
    
    # LLM Type
    MUTATION: str = "claude" # "openrouter" or "codegen-350M-mono" or "gemini" or "claude"

    GENERATIONS: int = 20 
    NET_TRAINING_EPOCHS: int = 10
    TEMPERATURE: float = 0.6

    DEVICE: str = "cuda" # options: ["cuda", "cpu", "both"]
    
    # MAP-ELITES
    DIM_MAP: int = 2
    N_NICHES: int = 25

    # Training Parameters
    LEARNING_RATE: float = 0.01  # Learning rate for network training
    BATCH_SIZE: int = 128  # Batch size for training
    MOMENTUM: float = 0.9  # SGD momentum
    WEIGHT_DECAY: float = 1e-4  # Weight decay for regularization
    
    # Network Architecture
    NUM_LAYERS: int = 18  # ResNet depth
    IMG_CHANNELS: int = 3  # Input channels (RGB)
    IMG_WIDTH: int = 256 # ViTacTip
    IMG_HEIGHT: int = 256 # ViTacTip
    
    # Task-specific configurations with defaults
    NUM_CLASSES: int = 7  # grating classes 
    OUTPUT_SIZE: int = 3   # Default output size for regression tasks
    
    # Temperature and Randomness
    TEMP_MIN: float = 0.1  # Minimum temperature for LLM generation
    TEMP_MAX: float = 0.9  # Maximum temperature for LLM generation

    # Constants for Archive Management
    BOUNDS_PADDING_PERCENT: float = 0.05
    DEFAULT_NORMALIZED_VALUE: float = 0.5
    DEFAULT_DESCRIPTOR_VALUE: float = 0.0
    
    # Cache Configuration
    USE_ARCHIVE_CACHE: bool = True  # Enable/disable archive caching
    CACHE_EXPIRY_DAYS: int = 30    # Days before cache expires
    FORCE_CACHE_REFRESH: bool = False  # Force regeneration of cache
    
    # Constants for Curiosity Score
    BASE_CURIOSITY_SCORE: float = 0.5
    SUCCESSFUL_CURIOSITY_BONUS: float = 1.0
    MIN_PADDING: float = 0.01 # Minimum padding value
    
    # API Keys for LLM Services
    GEMINI_API_KEY: str = ""  # Gemini API key (can be set in config or environment variable)
    ANTHROPIC_API_KEY: str = ""  # Anthropic API key (can be set in config or environment variable)


    # Directories for pre-generated networks 
    PREGEN_DIR_SINGLE_LABEL = "pregens/single_label_pregens"
    PREGEN_DIR_IMAGE_REGRESSION = "pregens/image_regression_pregens"
    PREGEN_DIR_RESNET = "pregens/resnet_pregens"


    CODEGEN_SAVE_DIR: str = "./"

    # The get_task_config method can be made "private" by convention (leading underscore)
    # as it's now an internal helper method.
    def _get_task_config(self) -> dict:
        """Get task-specific configuration based on TASK_TYPE."""
        task_configs = {
            "CIFAR-10": {"NUM_CLASSES": 10, "OUTPUT_SIZE": None},
            "shape": {"NUM_CLASSES": 21, "OUTPUT_SIZE": None},
            "grating": {"NUM_CLASSES": 7, "OUTPUT_SIZE": None},
            "force": {"NUM_CLASSES": None, "OUTPUT_SIZE": 3},
            "pose": {"NUM_CLASSES": None, "OUTPUT_SIZE": 2}
        }
        # Return the specific config dict, or an empty one if not found
        return task_configs.get(self.TASK_TYPE, {})

    # This method is called automatically after the object is created
    def __post_init__(self):
        """
        Post-initialization hook to update task-specific parameters.
        This ensures the config object is always consistent after creation.
        """
        # Get the task-specific parameters
        task_params = self._get_task_config()

        # Update the instance attributes if new values were found
        # The `get` method with a default value handles cases where a key
        # might not be in the returned dict (e.g. NUM_CLASSES for regression task)
        self.NUM_CLASSES = task_params.get("NUM_CLASSES", self.NUM_CLASSES)
        self.OUTPUT_SIZE = task_params.get("OUTPUT_SIZE", self.OUTPUT_SIZE)
        
        # Check for API keys in environment variables if not set in config
        if not self.GEMINI_API_KEY:
            import os
            env_gemini_key = os.getenv("GEMINI_API_KEY")
            if env_gemini_key:
                self.GEMINI_API_KEY = env_gemini_key
                
        if not self.ANTHROPIC_API_KEY:
            import os
            env_anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            if env_anthropic_key:
                self.ANTHROPIC_API_KEY = env_anthropic_key

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)