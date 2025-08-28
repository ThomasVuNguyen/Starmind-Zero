#!/usr/bin/env python3
"""
Inference script for pico-lm/pico-decoder-tiny model using Hugging Face transformers.
Optimized for clean, readable code and efficient inference on resource-constrained devices.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from typing import Optional, Dict, Any


class PicoLMInference:
    """Clean and efficient inference wrapper for pico-lm/pico-decoder-tiny model."""
    
    def __init__(self, model_name: str = "pico-lm/pico-decoder-tiny", device: Optional[str] = None):
        """
        Initialize the PicoLM inference engine.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run inference on ('cpu', 'cuda', or None for auto-detection)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model: {model_name}")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Model loaded successfully!")
    
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


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="PicoLM inference script")
    parser.add_argument("--prompt", "-p", type=str, help="Input prompt for text generation")
    parser.add_argument("--max-length", "-l", type=int, default=100, help="Maximum generation length")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", "-d", type=str, choices=["cpu", "cuda"], help="Device to use")
    
    args = parser.parse_args()
    
    try:
        # Initialize inference engine
        inference = PicoLMInference(device=args.device)
        
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
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
