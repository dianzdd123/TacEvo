import os
import logging
import torch
import argparse
import threading
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from flask import Flask, request, jsonify
import warnings

# Suppress specific transformer and user-level warnings
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LocalCodeGen:
    def __init__(self, model_dir=None, save_local=True, force_cpu=False):
        """
        Initialise the CodeGen model. Optionally save it locally for offline use.
        """
        #self.model_name = "Salesforce/codegen-350M-mono"
        #self.local_dir = model_dir or "./models/codegen-350M-mono"
        self.model_name = "Salesforce/codegen-350M-mono"
        self.local_dir = model_dir or "./models/codegen-350M-mono"
        
        # Device selection with CPU fallback option
        if force_cpu:
            self.device = torch.device("cpu")
            print("  Forcing CPU mode due to CUDA issues")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.save_local = save_local

        # Create models directory if it doesn't exist
        os.makedirs("./models", exist_ok=True)

        # Check for saved model files
        config_path = os.path.join(self.local_dir, "config.json")
        weight_file = None
        if os.path.exists(os.path.join(self.local_dir, "pytorch_model.bin")):
            weight_file = "pytorch_model.bin"
        elif os.path.exists(os.path.join(self.local_dir, "model.safetensors")):
            weight_file = "model.safetensors"

        model_files_exist = os.path.exists(config_path) and weight_file is not None

        if model_files_exist:
            logging.info(f"Loading model from local directory: {self.local_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.local_dir,
                use_safetensors=weight_file.endswith("safetensors")
            )
        else:
            logging.info(f"Model not found locally. Downloading: {self.model_name}")
            self._download_and_save_model()

        self.model.to(self.device)
        logging.info(f"Model loaded to device: {self.device}")

    def _download_and_save_model(self):
        """
        Helper to download and save model locally
        """
        logging.info(f"Downloading model from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            #use_safetensors=True,
            local_files_only=False
        )

        if self.save_local:
            os.makedirs(self.local_dir, exist_ok=True)  # Ensure directory exists
            logging.info(f"Saving model to {self.local_dir}")
            
            # Save tokenizer
            self.tokenizer.save_pretrained(self.local_dir)
            
            # Save model with explicit safetensors format
            self.model.save_pretrained(
                self.local_dir,
                use_safetensors=True,
                safe_serialization=True
            )

            # Verify saved files
            try:
                saved_files = os.listdir(self.local_dir)
                logging.info(f"Model saved successfully. Files in {self.local_dir}:")
                for file in saved_files:
                    file_path = os.path.join(self.local_dir, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                    logging.info(f"- {file} ({file_size:.2f} MB)")
            except Exception as e:
                logging.error(f"Error verifying saved files: {e}")
                raise  # Re-raise the exception to ensure we know if saving failed


    def check_gpu_status(self):
        """
        Returns device and memory stats
        """
        return {
            "cuda_available": torch.cuda.is_available(),
            "device": str(self.device),
            "model_on_cuda": next(self.model.parameters()).is_cuda if torch.cuda.is_available() else False,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_memory_allocated_MB": torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else None,
            "cuda_memory_reserved_MB": torch.cuda.memory_reserved(0) / 1024**2 if torch.cuda.is_available() else None,
        }

    def generate(self, prompt, max_tokens=100, temperature=0.2):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        output_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            num_return_sequences=1,
            early_stopping=True,
            no_repeat_ngram_size=2,
            repetition_penalty=1.2
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_stream(self, prompt, max_tokens=100, temperature=0.7):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            inputs=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for token in streamer:
            yield token

app = Flask(__name__)
llm_instance = None  # Will be initialized once

@app.route("/generate", methods=["POST"])
def generate_api():
    global llm_instance
    if llm_instance is None:
        llm_instance = LocalCodeGen()

    data = request.get_json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 100)
    temperature = data.get("temperature", 0.7)

    output = llm_instance.generate(prompt, max_tokens, temperature)
    return jsonify({"response": output})

def run_cli():
    parser = argparse.ArgumentParser(description="Generate text/code using CodeGen locally.")
    parser.add_argument("prompt", type=str, nargs="?", help="Prompt to generate text from")
    parser.add_argument("--stream", action="store_true", help="Stream tokens as they are generated")
    parser.add_argument("--save-local", action="store_true", help="Save model for offline use")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU status and exit")
    args = parser.parse_args()

    llm = LocalCodeGen(save_local=args.save_local)

    if args.check_gpu:
        import pprint
        pprint.pprint(llm.check_gpu_status())
        return

    if args.prompt is None:
        print("Please provide a prompt or use --check-gpu.")
        return

    if args.stream:
        print("Streaming output:\n")
        for token in llm.generate_stream(args.prompt):
            print(token, end="", flush=True)
        print()
    else:
        output = llm.generate(args.prompt)
        print("Generated output:\n")
        print(output)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Check if run as CLI or Flask API
    import sys
    if len(sys.argv) > 1 and sys.argv[1] != "flask":
        run_cli()
    else:
        print("Starting Flask server at http://localhost:5000")
        app.run(port=5000)
