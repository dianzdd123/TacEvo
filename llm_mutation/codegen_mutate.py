from .local_llm.local_codegen import LocalCodeGen
from conf.config import Config
import os

#  Local Model Section 

__llm_instance = None
__llm_instance_cpu = None

def get_llm_instance(cfg: Config):
    """
    Initializes and returns a singleton instance of the LocalCodeGen model.
    """
    global __llm_instance
    if __llm_instance is None:
        print(" Initializing CodeGen Model (first time only) ")
        model_path = os.path.join(cfg.CODEGEN_SAVE_DIR, cfg.MUTATION)
        __llm_instance = LocalCodeGen(model_dir=model_path, save_local=True)
    
    return __llm_instance

def get_llm_instance_cpu(cfg: Config):
    """
    Initializes and returns a CPU-only instance of the LocalCodeGen model.
    """
    global __llm_instance_cpu
    if __llm_instance_cpu is None:
        print(" Initializing CodeGen Model on CPU (first time only) ")
        model_path = os.path.join(cfg.CODEGEN_SAVE_DIR, cfg.MUTATION)
        __llm_instance_cpu = LocalCodeGen(model_dir=model_path, save_local=True, force_cpu=True)
    
    return __llm_instance_cpu

def codegen_mutate(cfg: Config, prompt: str, temperature: float):
    """
    Uses a locally-installed CodeGen model to generate a mutation.
    """
    try:
        llm = get_llm_instance(cfg)
        output = llm.generate(prompt, max_tokens=500, temperature=temperature)
        return output
    except Exception as e:
        print(f"Error in codegen_mutate: {e}")
        # Try with CPU if CUDA fails
        try:
            print(f"Attempting CPU generation as fallback")
            llm_cpu = get_llm_instance_cpu(cfg)
            output = llm_cpu.generate(prompt, max_tokens=500, temperature=temperature)
            return output
        except Exception as e2:
            print(f"CPU generation also failed: {e2}")
            raise e  # Re-raise the original error