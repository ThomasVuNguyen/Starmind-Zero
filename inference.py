#!/usr/bin/env python3
"""
Inference script for pico-lm model using local checkpoint with the updated model architecture.
Optimized for clean, readable code and efficient inference on resource-constrained devices.
"""

import torch
import os
import sys
import shutil
from typing import Optional, Dict, Any
import argparse

# Add the pico-train src directory to the path so we can import our model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pico-train', 'src'))

# Clear model cache to ensure fresh model loading
def clear_model_cache():
    """Clear model cache directories to ensure fresh model loading."""
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        os.path.expanduser("~/.cache/torch"),
        os.path.expanduser("~/.cache/transformers"),
        os.path.expanduser("~/.cache/safetensors")
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                print(f"Clearing cache: {cache_dir}")
                shutil.rmtree(cache_dir)
                print(f"✓ Cleared: {cache_dir}")
            except Exception as e:
                print(f"Warning: Could not clear {cache_dir}: {e}")

# Clear cache at module import
clear_model_cache()

class PicoLMInference:
    """Clean and efficient inference wrapper for pico-lm model using local checkpoint."""
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the PicoLM inference engine.
        
        Args:
            checkpoint_path: Path to the local checkpoint directory
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.checkpoint_path = checkpoint_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        print(f"Using device: {self.device}")
        
        # Import our model classes
        from model.pico_decoder import PicoDecoderForCausalLM, PicoDecoderHFConfig
        
        # Load config
        config_path = os.path.join(checkpoint_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config object
        self.config = PicoDecoderHFConfig(**config_dict)
        print(f"✓ Loaded config: vocab_size={self.config.vocab_size}")
        
        # Create model instance
        print("Creating model instance...")
        self.model = PicoDecoderForCausalLM(self.config)
        
        # Load weights from checkpoint
        print("Loading weights from checkpoint...")
        checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
        
        if os.path.exists(checkpoint_file):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file)
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)
            print("✓ Weights loaded successfully")
        else:
            raise FileNotFoundError(f"Model weights file not found: {checkpoint_file}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load tokenizer
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("✓ Model loaded successfully!")
    
    def generate_completion(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text completion for the given prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling vs greedy decoding
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text completion
        """
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate completion
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the output
        completion = generated_text[len(prompt):].strip()
        return completion
    
    def interactive_mode(self):
        """Run interactive chat mode for testing the model."""
        print("\n=== PicoLM Interactive Mode ===")
        print("Type 'quit' to exit, 'clear' to clear context")
        print("-" * 40)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    print("Context cleared.")
                    continue
                elif not user_input:
                    continue
                
                # Generate response
                print("Generating...")
                response = self.generate_completion(
                    user_input,
                    max_length=150,
                    temperature=0.8
                )
                
                print(f"PicoLM: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def build_checkpoint_path(model_name: str, step_number: int, base_dir: str = "pico-train/runs") -> str:
    """
    Build checkpoint path from model name and step number.
    
    Args:
        model_name: Name of the model (e.g., 'pico-decoder-tiny-dolma5M-v1')
        step_number: Checkpoint step number
        base_dir: Base directory for model runs
        
    Returns:
        Full path to the checkpoint directory
    """
    checkpoint_path = os.path.join(base_dir, model_name, "checkpoints", f"step_{step_number}")
    return checkpoint_path


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="PicoLM inference script using local checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py pico-decoder-tiny-dolma5M-v1 1000
  python inference.py pico-decoder-tiny-dolma5M-v1 1000 --interactive
  python inference.py pico-decoder-tiny-dolma5M-v1 1000 --prompt "Hello world"
  python inference.py --checkpoint path/to/checkpoint  # Use full path (legacy mode)
        """
    )
    
    # Positional arguments for simplified usage
    parser.add_argument("model_name", nargs="?", type=str, 
                       help="Model name (e.g., 'pico-decoder-tiny-dolma5M-v1')")
    parser.add_argument("step_number", nargs="?", type=int,
                       help="Checkpoint step number (e.g., 1000)")
    
    # Optional arguments
    parser.add_argument("--checkpoint", "-c", type=str, 
                       help="Full path to checkpoint directory (overrides model_name/step_number)")
    parser.add_argument("--prompt", "-p", type=str, help="Input prompt for text generation")
    parser.add_argument("--max-length", "-l", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    checkpoint_path = None
    
    if args.checkpoint:
        # Use explicitly provided checkpoint path (legacy mode)
        checkpoint_path = args.checkpoint
    elif args.model_name and args.step_number is not None:
        # Build checkpoint path from model name and step number
        checkpoint_path = build_checkpoint_path(args.model_name, args.step_number)
    else:
        # No valid arguments provided
        parser.error("Either provide model_name and step_number, or use --checkpoint with full path")
    
    # Validate checkpoint path exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        return 1
    
    config_path = os.path.join(checkpoint_path, "config.json")
    model_file = os.path.join(checkpoint_path, "model.safetensors")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        return 1
        
    if not os.path.exists(model_file):
        print(f"Error: Model file not found: {model_file}")
        return 1
    
    print(f"Using checkpoint: {checkpoint_path}")
    
    try:
        # Initialize inference engine
        inference = PicoLMInference(checkpoint_path=checkpoint_path, device=args.device)
        
        if args.interactive:
            inference.interactive_mode()
        elif args.prompt:
            # Generate single completion
            completion = inference.generate_completion(
                args.prompt,
                max_length=args.max_length,
                temperature=args.temperature
            )
            print(f"\nPrompt: {args.prompt}")
            print(f"Completion: {completion}")
        else:
            # Default interactive mode if no arguments provided
            print("No prompt provided. Starting interactive mode...")
            inference.interactive_mode()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
