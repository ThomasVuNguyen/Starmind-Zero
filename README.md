# Starmind-Zero
Our goal is to create a tiny device that runs smart AI fast!

## Getting Started

Clone this repository with submodules:
```bash
git clone --recurse-submodules https://github.com/ThomasVuNguyen/Starmind-Zero.git
```

## Updating Submodules

To update submodules to their latest commits:
```bash
git submodule update --remote
```

## Using the AI Model (inference.py)

The `inference.py` script provides a clean and efficient way to run inference with the pico-lm model using local checkpoints.

### Prerequisites

Make sure you have the required dependencies installed:
```bash
pip install torch transformers safetensors
```

### Basic Usage

The script supports two ways to specify checkpoints:

#### 1. Simplified Usage (Recommended)
Use model name and step number for easy access to local checkpoints:
```bash
python inference.py [model_name] [step_number]
```

#### 2. Full Path Usage (Legacy)
Specify the complete checkpoint path:
```bash
python inference.py --checkpoint path/to/checkpoint
```

### Quick Start Examples

#### Interactive Mode
Run the model in interactive chat mode:
```bash
python inference.py pico-decoder-tiny-dolma5M-v1 1000 --interactive
```

#### Single Prompt Generation
Generate text completion for a specific prompt:
```bash
python inference.py pico-decoder-tiny-dolma5M-v1 1000 --prompt "Once upon a time"
```

#### Creative Writing
Use higher temperature for more creative outputs:
```bash
python inference.py pico-decoder-tiny-dolma5M-v1 1000 --prompt "In a distant galaxy" --temperature 0.9 --max-length 200
```

### Command Line Options

**Positional Arguments:**
- `model_name`: Model name (e.g., 'pico-decoder-tiny-dolma5M-v1')
- `step_number`: Checkpoint step number (e.g., 1000)

**Optional Arguments:**
- `--checkpoint, -c`: Full path to checkpoint directory (overrides model_name/step_number)
- `--prompt, -p`: Input prompt for text generation
- `--max-length, -l`: Maximum generation length (default: 100)
- `--temperature, -t`: Sampling temperature (default: 0.7)
- `--interactive, -i`: Run in interactive mode
- `--device, -d`: Device to use (`cpu` or `cuda`)

### Available Models

Check the `pico-train/runs/` directory for available models:
- `pico-decoder-tiny-dolma5M-v1`
- `pico-decoder-tiny-dolma29k-v1`
- `pico-decoder-tiny-dolma29k-v2`
- `pico-decoder-tiny-dolma29k-v3`
- `pico-decoder-tiny-dolma-teensy-v0`
- `pico-decoder-tiny-dolma-teensy-v1`

### More Examples

1. **Quick test with latest checkpoint:**
   ```bash
   python inference.py pico-decoder-tiny-dolma5M-v1 32500 --prompt "Hello, how are you?"
   ```

2. **Force CPU usage:**
   ```bash
   python inference.py pico-decoder-tiny-dolma5M-v1 1000 --device cpu --prompt "Your prompt"
   ```

3. **Interactive mode with specific model:**
   ```bash
   python inference.py pico-decoder-tiny-dolma29k-v3 5000 --interactive
   ```

4. **Legacy mode with full path:**
   ```bash
   python inference.py --checkpoint pico-train/runs/pico-decoder-tiny-dolma5M-v1/checkpoints/step_1000 --interactive
   ```

### Interactive Mode Commands

When running in interactive mode:
- Type your message and press Enter to generate a response
- Type `clear` to clear the conversation context
- Type `quit` to exit the program

### Troubleshooting

- **Model not found**: Ensure the checkpoint path is correct and contains `config.json` and `model.safetensors` files
- **CUDA out of memory**: Use `--device cpu` to force CPU usage
- **Import errors**: Make sure you're running from the repository root directory
