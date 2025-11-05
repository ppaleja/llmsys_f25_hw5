# Problem 1.3: Performance Comparison

This directory contains tools for comparing performance between single device and multi-device distributed training.

## Files

- **`problem_1_3_notebook.ipynb`**: Jupyter notebook for running experiments in Modal's notebook environment
- **`problem_1_3_analysis.py`**: Standalone Python script for analyzing metrics from existing runs
- **`submit/figures/`**: Directory where output figures are saved

## Recommended Approach

We recommend using **Option 2** (the standalone analysis script) for most users, as it's simpler and works locally or on PSC machines. Use Option 1 if you specifically need to run everything in Modal's cloud infrastructure.

## Option 1: Using the Jupyter Notebook (For Modal Cloud Environment)

The Jupyter notebook (`problem_1_3_notebook.ipynb`) is designed to run in Modal's notebook environment or Modal app. It will:

1. Set up Modal infrastructure
2. Run single device training (world_size=1, batch_size=64)
3. Run multi-device training (world_size=2, batch_size=128)
4. Collect metrics (excluding first epoch as warmup)
5. Generate comparison plots
6. Save figures to `submit/figures/`

### Prerequisites:

```bash
pip install modal
modal token new  # Authenticate with Modal
```

### Usage:

1. Upload the project to Modal or run in Modal's notebook environment
2. Open the notebook (`problem_1_3_notebook.ipynb`)
3. Run all cells sequentially
4. The notebook will:
   - Set up Modal infrastructure
   - Execute training runs (this may take 1-2 hours depending on GPU availability)
   - Collect metrics automatically
   - Generate and display plots
5. Figures are saved to `submit/figures/`

**Note:** This approach requires Modal credits and may incur costs.

## Option 2: Using the Analysis Script (For Local/PSC Runs)

If you run training experiments locally or on PSC machines, you can use the standalone script to analyze the results:

### Step 1: Run Single Device Training

```bash
python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 10
```

This will create metric files in `./workdir/`:
- `rank0_results_epoch0.json` through `rank0_results_epoch9.json`

### Step 2: Run Multi-Device Training

```bash
python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 10
```

This will create metric files in `./workdir/`:
- `rank0_results_epoch0.json` through `rank0_results_epoch9.json`
- `rank1_results_epoch0.json` through `rank1_results_epoch9.json`

### Step 3: Analyze Results

```bash
python problem_1_3_analysis.py --workdir ./workdir --n-epochs 10
```

This will:
- Load metrics from the workdir
- Exclude the first epoch (warmup)
- Calculate average training time and tokens per second
- Generate comparison plots
- Save figures to `submit/figures/`

### Command-line Options:

```bash
python problem_1_3_analysis.py --help

Options:
  --workdir WORKDIR              Path to workdir containing result JSON files (default: ./workdir)
  --single-world-size SIZE       World size for single device run (default: 1)
  --multi-world-size SIZE        World size for multi-device run (default: 2)
  --n-epochs N                   Number of epochs (default: 10)
  --output-dir DIR               Output directory for figures (default: submit/figures)
```

## Metrics Collected

### Training Time
- **Definition**: Time taken to complete one epoch of training
- **Aggregation**: Average across devices, then average across epochs (excluding first epoch)
- **Lower is better**

### Tokens Per Second (Throughput)
- **Definition**: Number of tokens processed per second
- **Aggregation**: Sum across devices (total throughput), then average across epochs (excluding first epoch)
- **Higher is better**

## Output Figures

The analysis generates two figures:

1. **`training_time_comparison.png`**: Bar chart comparing training time between single and multi-device
2. **`tokens_per_second_comparison.png`**: Bar chart comparing throughput between single and multi-device

Both figures include:
- Mean values (bar height)
- Standard deviation (error bars)
- Numerical values on each bar

## Example Output

```
==============================================================
Problem 1.3: Performance Comparison Analysis
==============================================================

Loading single device metrics (world_size=1)...
Single Device Metrics (excluding first epoch):
  Training Time: 45.23 ± 2.15 seconds
  Tokens/Second: 2845.67 ± 123.45

Loading multi-device metrics (world_size=2)...
Multi-Device Metrics (excluding first epoch):
  Training Time: 24.56 ± 1.89 seconds
  Tokens/Second: 5234.89 ± 234.56

Speedup:
  Training Time Speedup: 1.84x
  Throughput Speedup: 1.84x

Generating plots...
Saved figure to submit/figures/training_time_comparison.png
Saved figure to submit/figures/tokens_per_second_comparison.png

==============================================================
SUMMARY
==============================================================
Figures saved to: submit/figures/
  - training_time_comparison.png
  - tokens_per_second_comparison.png
==============================================================
```

## Notes

- The first epoch is excluded as a warmup to get more stable measurements
- Training time is averaged across devices for each epoch
- Tokens per second is summed across devices (total system throughput)
- Results are then averaged across epochs to account for variability
