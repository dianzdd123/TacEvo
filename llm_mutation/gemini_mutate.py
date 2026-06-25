import google.generativeai as genai
import os
from conf.config import Config

cfg = Config()

def gemini_mutate(prompt: str, temperature: float, model="gemma-3-27b-it", cfg: Config = None): # TODO: removed the first arg (cfg) since its is not used, so make sure it gets called correctly
    """
    Uses the Gemini API to generate a mutation.
    """
    # Use provided config or create default instance
    if cfg is None:
        cfg = Config()
    
    # Load API key from config first, then fall back to environment variable
    api_key = cfg.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in config or environment variable GEMINI_API_KEY.")

    # Configure Gemini
    genai.configure(api_key=api_key)

    try:
        # Create model instance
        model_instance = genai.GenerativeModel(model)
        
        # Generate response
        response = model_instance.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=4000,
            )
        )

        return response.text
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return ""  # Return an empty string on failure