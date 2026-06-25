import os
import anthropic
from conf.config import Config


def claude_mutate(prompt: str, temperature: float, model="claude-3-haiku-20240307", cfg: Config = None):
    
    # Use provided config or create default instance
    if cfg is None:
        cfg = Config()
    
    # Load API key from config first, then fall back to environment variable
    api_key = cfg.ANTHROPIC_API_KEY or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set in config or environment variable ANTHROPIC_API_KEY.")
    
    # configure anthropic
    client = anthropic.Anthropic(api_key=api_key)
    
    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the content from the response
        generated_code = response.content[0].text

        return generated_code
    
    except Exception as e:
        print(f"Error generating network with Claude: {e}")
        return "" # Return an empty string on failure