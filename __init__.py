"""
Tactile NAS - Neural Architecture Search for Tactile Sensing
"""
# Version information
__version__ = "1.0.0"

# Import commonly used functions at package level for convenience
from .core.utils import read_python_file, extract_and_validate_code, is_trainable, get_main_network_class, csv_writer

# Make these available when someone does: from tactile_nas import read_python_file
__all__ = [
    'read_python_file',
    'extract_and_validate_code', 
    'is_trainable',
    'get_main_network_class',
    'csv_writer'
]
