"""
OrwellBot Modal deployment - Serverless GPU inference.

This deploys OrwellBot (Qwen2.5-1.5B-Instruct + LoRA adapters) as a
serverless endpoint on Modal with GPU acceleration.

Usage:
    modal deploy deployment/modal_app.py

Then test with:
    curl -X POST https://<your-app>.modal.run/generate_text \
         -H "Content-Type: application/json" \
         -d '{"prompt": "Power is", "max_tokens": 100}'
"""

import modal

# Modal app configuration
app = modal.App("orwellbot")

# Volume for storing PEFT adapters
adapters_volume = modal.Volume.from_name("orwellbot-adapters", create_if_missing=True)

# Container image with required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "peft>=0.7.0",
        "safetensors>=0.4.0",
        "accelerate>=0.25.0",
        "bitsandbytes>=0.41.0",
        "fastapi",
    )
)

# Model configuration
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "/orwellbot"  # Path in the volume


@app.cls(
    image=image,
    gpu="T4",
    volumes={ADAPTER_PATH: adapters_volume},
    timeout=300,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=5)
class OrwellBot:
    """OrwellBot inference class with GPU acceleration."""

    @modal.enter()
    def load_model(self):
        """Load model and adapters on container startup."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading base model: {BASE_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        # Load base model with 4-bit quantization for memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load PEFT adapters if available
        adapter_config_path = f"{ADAPTER_PATH}/adapter_config.json"
        try:
            import os
            if os.path.exists(adapter_config_path):
                print(f"Loading PEFT adapters from: {ADAPTER_PATH}")
                self.model = PeftModel.from_pretrained(
                    self.model,
                    ADAPTER_PATH,
                    is_trainable=False,
                )
                print("Adapters loaded successfully!")
            else:
                print("Warning: No adapters found, using base model")
        except Exception as e:
            print(f"Warning: Could not load adapters: {e}")
            print("Using base model without fine-tuning")

        self.model.eval()
        print("Model ready for inference!")

    @modal.method()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
    ) -> dict:
        """Generate text from a prompt."""
        import torch

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return {
            "prompt": prompt,
            "response": response.strip(),
            "tokens_generated": len(generated_tokens),
        }


@app.function(image=image, timeout=300)
@modal.fastapi_endpoint(method="POST")
def generate_text(request: dict) -> dict:
    """
    Web endpoint for text generation.

    Request body:
        {
            "prompt": str,           # Required: the text prompt
            "max_tokens": int,       # Optional: max tokens to generate (default: 200)
            "temperature": float,    # Optional: sampling temperature (default: 0.7)
            "top_p": float,          # Optional: nucleus sampling (default: 0.9)
            "repetition_penalty": float  # Optional: repetition penalty (default: 1.1)
        }

    Response:
        {
            "prompt": str,
            "response": str,
            "tokens_generated": int
        }
    """
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "No prompt provided"}

    max_tokens = request.get("max_tokens", 200)
    temperature = request.get("temperature", 0.7)
    top_p = request.get("top_p", 0.9)
    repetition_penalty = request.get("repetition_penalty", 1.1)

    # Validate parameters
    max_tokens = min(max(1, max_tokens), 500)  # Clamp to 1-500
    temperature = min(max(0.1, temperature), 2.0)  # Clamp to 0.1-2.0
    top_p = min(max(0.1, top_p), 1.0)  # Clamp to 0.1-1.0
    repetition_penalty = min(max(1.0, repetition_penalty), 2.0)  # Clamp to 1.0-2.0

    bot = OrwellBot()
    return bot.generate.remote(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )


@app.function(image=image)
@modal.fastapi_endpoint(method="GET")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "model": BASE_MODEL}


# Local testing
if __name__ == "__main__":
    # For local testing with `modal run`
    with app.run():
        result = generate_text.remote({
            "prompt": "The greatest enemy of clear language is",
            "max_tokens": 100,
            "temperature": 0.7,
        })
        print(result)
