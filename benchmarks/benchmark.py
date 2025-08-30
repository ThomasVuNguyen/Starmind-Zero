#!/usr/bin/env python3
"""
Mass evaluation script for running predefined prompts through all checkpoints of a model.
Simple, clean, and minimal approach with readable markdown logging.
"""

import os
import sys
import glob
import time
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the parent directory to path so we can import inference
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def load_prompts(prompts_file: str = "prompts.json") -> List[str]:
    """
    Load benchmark prompts from JSON file.
    
    Args:
        prompts_file: Path to the prompts JSON file
        
    Returns:
        List of prompt strings
    """
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_path = os.path.join(script_dir, prompts_file)
    
    if not os.path.exists(prompts_path):
        print(f"⚠️  Prompts file not found: {prompts_path}")
        print("Using default fallback prompts...")
        # Fallback prompts if file doesn't exist
        return ["Hello, how are you?"]
    
    try:
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        # Handle both old format (dict with benchmark_prompts) and new format (simple list)
        if isinstance(prompts, dict) and "benchmark_prompts" in prompts:
            # Old format - extract text field
            prompts = [p.get("text", str(p)) for p in prompts["benchmark_prompts"]]
        elif isinstance(prompts, list):
            # New simple format - already a list of strings
            pass
        else:
            print("⚠️  Invalid prompts format, using fallback")
            return ["Hello, how are you?"]
        
        print(f"📝 Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts
        
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing prompts file: {e}")
        print("Using default fallback prompts...")
        return ["Hello, how are you?"]
    except Exception as e:
        print(f"❌ Error loading prompts file: {e}")
        print("Using default fallback prompts...")
        return ["Hello, how are you?"]


def discover_checkpoints(model_name: str, base_dir: str = "../pico-train/runs") -> List[str]:
    """
    Discover all available checkpoints for a given model.
    
    Args:
        model_name: Name of the model
        base_dir: Base directory for model runs
        
    Returns:
        List of checkpoint paths sorted by step number
    """
    model_path = os.path.join(base_dir, model_name, "checkpoints")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    
    # Find all step_* directories
    pattern = os.path.join(model_path, "step_*")
    checkpoint_dirs = glob.glob(pattern)
    
    # Filter out non-directories and extract step numbers for sorting
    valid_checkpoints = []
    for checkpoint_dir in checkpoint_dirs:
        if os.path.isdir(checkpoint_dir):
            try:
                step_num = int(os.path.basename(checkpoint_dir).split('_')[1])
                valid_checkpoints.append((step_num, checkpoint_dir))
            except (IndexError, ValueError):
                continue
    
    # Sort by step number and return paths
    valid_checkpoints.sort(key=lambda x: x[0])
    return [checkpoint_path for _, checkpoint_path in valid_checkpoints]


def run_benchmark(model_name: str, output_dir: str = "results", prompts_file: str = "prompts.json") -> str:
    """
    Run benchmark evaluation on all checkpoints of a model.
    
    Args:
        model_name: Name of the model to benchmark
        output_dir: Directory to save results
        prompts_file: Path to the prompts JSON file
        
    Returns:
        Path to the generated report file
    """
    print(f"🚀 Starting benchmark for model: {model_name}")
    
    # Load prompts
    benchmark_prompts = load_prompts(prompts_file)
    if not benchmark_prompts:
        print("❌ No prompts loaded")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Discover checkpoints
    try:
        checkpoints = discover_checkpoints(model_name)
        print(f"📊 Found {len(checkpoints)} checkpoints")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None
    
    if not checkpoints:
        print("❌ No valid checkpoints found")
        return None
    
    # Generate report filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f"{model_name}_benchmark_{timestamp}.md")
    
    # Import inference module
    try:
        from inference import PicoLMInference
    except ImportError as e:
        print(f"❌ Failed to import inference module: {e}")
        return None
    
    # Start writing report
    with open(report_file, 'w') as f:
        f.write(f"# Benchmark Report: {model_name}\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Checkpoints**: {len(checkpoints)}\n")
        f.write(f"**Total Prompts**: {len(benchmark_prompts)}\n\n")
        f.write("---\n\n")
        
        # Process each checkpoint
        for i, checkpoint_path in enumerate(checkpoints, 1):
            checkpoint_name = os.path.basename(checkpoint_path)
            print(f"📝 Processing {checkpoint_name} ({i}/{len(checkpoints)})")
            
            f.write(f"## Checkpoint: {checkpoint_name}\n\n")
            f.write(f"**Path**: `{checkpoint_path}`\n\n")
            
            try:
                # Load model for this checkpoint
                start_time = time.time()
                inference = PicoLMInference(checkpoint_path=checkpoint_path, device="cuda")
                load_time = time.time() - start_time
                
                f.write(f"**Load Time**: {load_time:.2f}s\n\n")
                
                # Run all prompts
                for j, prompt_text in enumerate(benchmark_prompts, 1):
                    print(f"  └─ Prompt {j}/{len(benchmark_prompts)}: {prompt_text[:30]}...")
                    
                    f.write(f"### Prompt {j}: \"{prompt_text}\"\n\n")
                    
                    try:
                        # Generate response with default parameters
                        gen_start = time.time()
                        response = inference.generate_completion(
                            prompt=prompt_text,
                            max_length=100,
                            temperature=0.7
                        )
                        gen_time = time.time() - gen_start
                        
                        f.write(f"**Response**:\n```\n{response}\n```\n\n")
                        f.write(f"**Metadata**: max_length=100, temperature=0.7, time={gen_time:.2f}s\n\n")
                        
                    except Exception as e:
                        f.write(f"**Error**: {str(e)}\n\n")
                        print(f"    ⚠️  Error on prompt {j}: {e}")
                
            except Exception as e:
                f.write(f"**Checkpoint Error**: {str(e)}\n\n")
                print(f"  ❌ Failed to load checkpoint: {e}")
            
            f.write("---\n\n")
    
    print(f"✅ Benchmark complete! Report saved to: {report_file}")
    return report_file


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on all checkpoints of a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py pico-decoder-tiny-dolma5M-v1
  python benchmark.py pico-decoder-tiny-dolma29k-v3 --output results/
        """
    )
    
    parser.add_argument("model_name", type=str, 
                       help="Model name (e.g., 'pico-decoder-tiny-dolma5M-v1')")
    parser.add_argument("--output", "-o", type=str, default="results",
                       help="Output directory for results (default: results)")
    parser.add_argument("--prompts", "-p", type=str, default="prompts.json",
                       help="Prompts JSON file (default: prompts.json)")
    
    args = parser.parse_args()
    
    try:
        report_file = run_benchmark(args.model_name, args.output, args.prompts)
        if report_file:
            print(f"\n📄 Report available at: {report_file}")
            return 0
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
