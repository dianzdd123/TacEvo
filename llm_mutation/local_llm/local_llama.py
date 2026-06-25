import os
import logging
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class LocalLlama:
    def __init__(self, model_path=None):
        """
        Initialize the Llama model. If no model path is given, download it from Hugging Face.
        """
        self.model_path = model_path or self._get_or_download_model()
        try:
            # Check if CUDA is available
            import torch
            if torch.cuda.is_available():
                logging.info("CUDA is available, using GPU acceleration")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=16,
                    use_mmap=False,  # Disable memory mapping for GPU
                    use_mlock=True   # Lock memory to prevent swapping
                )
            else:
                logging.warning("CUDA not available, falling back to CPU")
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=4,
                    n_gpu_layers=0
                )
        except ImportError:
            logging.warning("CUDA not available, falling back to CPU")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                n_gpu_layers=0
            )
        logging.info(f"Llama model loaded from: {self.model_path}")

    def _get_or_download_model(self):
        """
        Download model from Hugging Face if not already present.
        Returns the local path to the model.
        """
        model_name = "llama-2-7b-chat.Q4_K_M.gguf"
        local_path = os.path.join("models", model_name)
        os.makedirs("models", exist_ok=True)

        if not os.path.exists(local_path):
            logging.info("Downloading model from Hugging Face")
            hf_hub_download(
                repo_id="TheBloke/Llama-2-7B-Chat-GGUF",
                filename=model_name,
                local_dir="models",
                local_dir_use_symlinks=False
            )
            logging.info("Model download complete.")

        return local_path

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """
        Generate text using the Llama model
        """
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        response = self.llm(
            formatted_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "\n\n"],
            echo=False
        )
        return response["choices"][0]["text"].strip()

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = LocalLlama()
    response = llm.generate("What is the capital of France?")
    print(response)
