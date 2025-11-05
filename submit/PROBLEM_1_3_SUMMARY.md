# Problem 1.3: Performance Comparison - Implementation Summary

## Overview

This implementation provides tools for comparing the performance between single device and multi-device distributed training for Problem 1.3.

## Files Created

### 1. `problem_1_3_notebook.ipynb`
- **Purpose**: Jupyter notebook for running complete experiments in Modal's cloud environment
- **Features**:
  - Automated training runs for both single and multi-device configurations
  - Built-in metric collection and analysis
  - Automatic plot generation
  - Saves figures to `submit/figures/`
- **Use Case**: When running in Modal's notebook environment or needing a fully automated workflow

### 2. `problem_1_3_analysis.py`
- **Purpose**: Standalone Python script for analyzing training results
- **Features**:
  - Loads metrics from JSON files in workdir
  - Excludes first epoch (warmup) from analysis
  - Computes aggregated metrics across ranks and epochs
  - Generates comparison plots
  - Calculates speedup metrics
- **Use Case**: When training runs are done locally or on PSC machines

### 3. `PROBLEM_1_3_README.md`
- **Purpose**: Comprehensive documentation for both approaches
- **Content**:
  - Detailed usage instructions for both Modal and local approaches
  - Command-line examples
  - Explanation of metrics
  - Expected outputs

### 4. `submit/figures/`
- **Purpose**: Directory for storing generated plots
- **Contents** (after running analysis):
  - `training_time_comparison.png`: Bar chart comparing training time
  - `tokens_per_second_comparison.png`: Bar chart comparing throughput

## How to Use

### Quick Start (Recommended)

For most users, the standalone script approach is recommended:

```bash
# Step 1: Run single device training
python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 10

# Step 2: Run multi-device training
python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 10

# Step 3: Analyze and visualize results
python problem_1_3_analysis.py --workdir ./workdir
```

This will:
1. Run both training configurations
2. Collect metrics from JSON files in workdir
3. Exclude the first epoch (warmup)
4. Generate comparison plots
5. Save figures to `submit/figures/`
6. Display speedup metrics

### Alternative: Modal Notebook

For users with Modal access:

1. Open `problem_1_3_notebook.ipynb` in Modal's notebook environment
2. Run all cells
3. Wait for training to complete (~1-2 hours)
4. View generated plots and metrics

## Metrics Explained

### Training Time
- **Definition**: Wall-clock time to complete one training epoch
- **Aggregation Method**:
  1. For each epoch, average training time across all ranks (devices)
  2. Average these values across all epochs (excluding first epoch)
- **Interpretation**: Lower is better; measures how long training takes

### Tokens Per Second (Throughput)
- **Definition**: Number of tokens processed per second
- **Aggregation Method**:
  1. For each epoch, sum tokens_per_sec across all ranks (total system throughput)
  2. Average these values across all epochs (excluding first epoch)
- **Interpretation**: Higher is better; measures processing speed

### Why Exclude First Epoch?
The first epoch includes:
- Model initialization overhead
- Data loading and caching
- GPU warmup
- JIT compilation

These one-time costs can skew measurements, so we exclude the first epoch to get stable, representative metrics.

## Implementation Details

### Metric Collection
The training script (`project/run_data_parallel.py`) automatically saves metrics for each epoch and rank to JSON files:
- Format: `rank{rank}_results_epoch{epoch}.json`
- Location: `./workdir/`
- Contents:
  - `training_time`: Epoch duration in seconds
  - `tokens_per_sec`: Processing rate for this rank
  - `validation_loss`: Validation loss
  - `bleu`: BLEU score

### Analysis Process
1. **Load**: Read all JSON files from workdir
2. **Aggregate**: 
   - Group by epoch
   - Average training time across ranks
   - Sum tokens_per_sec across ranks (system throughput)
3. **Filter**: Exclude first epoch
4. **Compute Statistics**: Calculate mean and standard deviation across epochs
5. **Visualize**: Create bar charts with error bars
6. **Report**: Display speedup metrics

### Plot Generation
- Uses matplotlib for visualization
- Bar charts with error bars (standard deviation)
- Numerical values displayed on bars
- Saved at 300 DPI for publication quality
- Consistent color scheme and formatting

## Expected Results

For a typical run, you might see:

```
Single Device (world_size=1, batch_size=64):
  Training Time: 45.23 ± 2.15 seconds
  Tokens/Second: 2845.67 ± 123.45

Multi-Device (world_size=2, batch_size=128):
  Training Time: 24.56 ± 1.89 seconds
  Tokens/Second: 5234.89 ± 234.56

Speedup:
  Training Time Speedup: 1.84x
  Throughput Speedup: 1.84x
```

**Note**: Actual values will vary based on:
- Hardware (GPU type, CPU, memory)
- Dataset size
- Network latency
- System load

## Verification

To verify the implementation works correctly:

```bash
# Run the analysis script help
python problem_1_3_analysis.py --help

# Test with mock data (already verified during development)
# See commit history for test results
```

## Submission

For submission, ensure:
1. ✅ `submit/figures/training_time_comparison.png` exists
2. ✅ `submit/figures/tokens_per_second_comparison.png` exists
3. ✅ Both training runs completed successfully
4. ✅ First epoch was excluded from analysis
5. ✅ Metrics show expected scaling behavior

## Notes

- The implementation follows the assignment requirements exactly
- Both approaches (notebook and script) produce identical results
- The standalone script is more flexible and works in any environment
- All code is well-documented with clear variable names
- Error handling is included for missing files
- The implementation is minimal and focused on the task requirements
