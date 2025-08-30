# Mass Evaluations

Simple benchmark tool for running predefined prompts through all checkpoints of a model.

## Usage

```bash
python benchmark.py [model_name] [options]
```

## Examples

```bash
# Benchmark all checkpoints of a model
python benchmark.py pico-decoder-tiny-dolma5M-v1

# Specify custom output directory
python benchmark.py pico-decoder-tiny-dolma5M-v1 --output my_results/

# Use custom prompts file
python benchmark.py pico-decoder-tiny-dolma5M-v1 --prompts my_prompts.json
```

## Managing Prompts

Prompts are stored in `prompts.json` as a simple array of strings:

```json
[
  "Hello, how are you?",
  "Complete this story: Once upon a time",
  "What is the capital of France?"
]
```

### Adding New Prompts

Simply edit `prompts.json` and add new prompt strings to the array. Super simple!

## Features

- **Auto-discovery**: Finds all `step_*` checkpoints automatically
- **JSON-based prompts**: Easily customizable prompts via JSON file
- **Readable output**: Markdown reports with clear structure
- **Error handling**: Continues on failures, logs errors
- **Progress tracking**: Shows real-time progress
- **Metadata logging**: Includes generation time and parameters

## Output

Results are saved as markdown files in `results/` directory:
```
results/
├── pico-decoder-tiny-dolma5M-v1_benchmark_20250101_120000.md
├── pico-decoder-tiny-dolma29k-v3_benchmark_20250101_130000.md
└── ...
```

## Predefined Prompts

1. "Hello, how are you?" (conversational)
2. "Complete this story: Once upon a time" (creative)
3. "Explain quantum physics in simple terms" (explanatory)
4. "Write a haiku about coding" (creative + structured)
5. "What is the capital of France?" (factual)
6. "The meaning of life is" (philosophical)
7. "In the year 2050," (futuristic)
8. "Python programming is" (technical)
