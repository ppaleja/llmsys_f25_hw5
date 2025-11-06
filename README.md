# Distributed Training and Parallelism for GPT-2

This project implements distributed training methods for GPT-2 language models, including **Data Parallelism** and **Pipeline Parallelism**. These techniques enable efficient training of large models across multiple GPUs by distributing computation and data processing.

## Overview

This project demonstrates two key distributed training techniques:

1. **Data Parallelism**: Distributes training data across multiple GPUs, with each GPU maintaining a complete copy of the model. Gradients are synchronized across GPUs to ensure consistent model updates.

2. **Pipeline Parallelism**: Divides the model layers across multiple GPUs, with each GPU responsible for a subset of layers. Input data flows through the pipeline in microbatches, allowing parallel computation across different model partitions.

## Features

### Data Parallel Training
- Dataset partitioning across multiple GPUs
- Gradient aggregation using `torch.distributed`
- Multi-process training with PyTorch multiprocessing
- Performance metrics (training time, tokens per second)
- Scalability visualization and benchmarking

### Pipeline Parallel Training
- Layer-wise model partitioning
- Microbatch scheduling for efficient pipeline execution
- Worker-based parallel computation
- GPT-2 model adaptation for pipeline parallelism
- Performance comparison with model parallelism

## Project Structure

```
llmsys_f25_hw5/
├── data_parallel/          # Data parallelism implementation
│   └── dataset.py          # Dataset partitioning utilities
├── pipeline/               # Pipeline parallelism implementation
│   ├── model.py           # Custom GPT-2 model definitions
│   ├── model_parallel.py  # Pipeline parallel model wrapper
│   ├── partition.py       # Module splitting utilities
│   ├── pipe.py           # Pipeline execution engine
│   └── worker.py         # Worker threads for parallel execution
├── project/               # Main training scripts
│   ├── run_data_parallel.py  # Data parallel training script
│   ├── run_pipeline.py       # Pipeline parallel training script
│   ├── utils.py              # Training utilities
│   └── plot.py               # Visualization utilities
├── tests/                 # Test suite
│   ├── test_data_parallel.py
│   └── test_pipeline.py
├── modal_run.py          # Modal deployment for data parallel
├── modal_run_pipeline.py # Modal deployment for pipeline parallel
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Installation

### Prerequisites
- Python 3.9+ (3.11 recommended for Modal deployment)
- CUDA-compatible GPUs (at least 2 GPUs for distributed training)
- PyTorch 2.2.0 with CUDA support

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ppaleja/llmsys_f25_hw5.git
cd llmsys_f25_hw5
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using the newer pyproject.toml:
```bash
pip install -e .
```

### Dependencies

Core dependencies include:
- `torch==2.2.0` - PyTorch deep learning framework
- `transformers==4.37.2` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `sacrebleu==2.4.0` - BLEU score evaluation
- `matplotlib` - Plotting and visualization
- `modal` - Cloud deployment (optional)

## Usage

### Data Parallel Training

Run data parallel training with single GPU:
```bash
python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 10
```

Run data parallel training with 2 GPUs:
```bash
python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 10
```

Run data parallel training with 4 GPUs:
```bash
python project/run_data_parallel.py --world_size 4 --batch_size 256 --n_epochs 10
```

### Pipeline Parallel Training

Run with model parallelism only:
```bash
python project/run_pipeline.py --model_parallel_mode='model_parallel'
```

Run with pipeline parallelism:
```bash
python project/run_pipeline.py --model_parallel_mode='pipeline_parallel'
```

### Modal Deployment (Cloud Training)

For cloud-based distributed training using Modal:

Data parallel training:
```bash
modal run modal_run.py --world-size 2 --n-epochs 5
```

Pipeline parallel training:
```bash
modal run modal_run_pipeline.py
```

## Testing

Run all tests:
```bash
python -m pytest -v
```

Test data parallel implementation:
```bash
python -m pytest -l -v -k "a5_1"
```

Test pipeline parallel implementation:
```bash
python -m pytest -l -v -k "a5_2"
```

## Performance Metrics

The training scripts automatically collect and report:
- **Training Time**: Average time per epoch (seconds)
- **Tokens Per Second**: Throughput metric for training efficiency

Results are saved in JSON format in the `workdir/` directory for further analysis.

## Visualization

Use the plotting utilities to visualize scaling performance:

```python
from project.plot import plot_results

# Plot training time comparison
plot_results(metrics_data, 'training_time')

# Plot throughput comparison
plot_results(metrics_data, 'tokens_per_second')
```

## Implementation Details

### Data Parallelism
- Uses `torch.distributed` for gradient synchronization
- Implements custom dataset partitioning to avoid data overlap
- Supports both NCCL (CUDA) and GLOO (CPU) backends
- Gradient averaging across all workers using all-reduce operations

### Pipeline Parallelism
- Implements microbatch scheduling for pipeline stages
- Uses worker threads for parallel computation across devices
- Handles tuple outputs from GPT-2 transformer blocks
- Optimizes for minimal pipeline bubbles

## License

This project is part of the LLM Systems course (Fall 2025).
