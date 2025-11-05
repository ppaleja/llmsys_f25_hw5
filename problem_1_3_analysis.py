"""
Problem 1.3: Performance Comparison Analysis Script

This script collects and analyzes metrics from distributed training runs,
comparing single device vs multi-device performance.

Usage:
    python problem_1_3_analysis.py --workdir ./workdir
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def compute_aggregated_metrics(metrics_by_rank, drop_first_epoch=True):
    """Compute aggregated metrics across ranks and epochs
    
    Args:
        metrics_by_rank: Dict mapping rank to list of epoch metrics
        drop_first_epoch: Whether to drop the first epoch (warmup)
    
    Returns:
        Dict with aggregated metrics
    """
    # Collect all training times and tokens_per_sec
    all_training_times = []
    all_tokens_per_sec_by_epoch = []
    
    world_size = len(metrics_by_rank)
    # Check if metrics_by_rank is non-empty and contains data before accessing
    if not metrics_by_rank or 0 not in metrics_by_rank:
        return {
            'avg_training_time': 0.0,
            'std_training_time': 0.0,
            'avg_tokens_per_sec': 0.0,
            'std_tokens_per_sec': 0.0,
            'world_size': world_size,
        }
    n_epochs = len(metrics_by_rank[0])
    
    start_epoch = 1 if drop_first_epoch else 0
    
    # For each epoch (excluding first if drop_first_epoch)
    for epoch_idx in range(start_epoch, n_epochs):
        epoch_training_times = []
        epoch_tokens_per_sec = []
        
        # Collect metrics from all ranks for this epoch
        for rank in range(world_size):
            if epoch_idx < len(metrics_by_rank[rank]):
                epoch_data = metrics_by_rank[rank][epoch_idx]
                epoch_training_times.append(epoch_data['training_time'])
                epoch_tokens_per_sec.append(epoch_data['tokens_per_sec'])
        
        # Average training time across ranks for this epoch
        all_training_times.append(np.mean(epoch_training_times))
        
        # Sum tokens_per_sec across ranks (total throughput) for this epoch
        all_tokens_per_sec_by_epoch.append(np.sum(epoch_tokens_per_sec))
    
    # Now compute mean and std across epochs
    # Check if we have any data to compute statistics
    if len(all_training_times) == 0:
        return {
            'avg_training_time': 0.0,
            'std_training_time': 0.0,
            'avg_tokens_per_sec': 0.0,
            'std_tokens_per_sec': 0.0,
            'world_size': world_size,
        }
    
    avg_training_time = np.mean(all_training_times)
    std_training_time = np.std(all_training_times)
    
    avg_tokens_per_sec = np.mean(all_tokens_per_sec_by_epoch)
    std_tokens_per_sec = np.std(all_tokens_per_sec_by_epoch)
    
    return {
        'avg_training_time': avg_training_time,
        'std_training_time': std_training_time,
        'avg_tokens_per_sec': avg_tokens_per_sec,
        'std_tokens_per_sec': std_tokens_per_sec,
        'world_size': world_size,
    }


def load_metrics_from_workdir(workdir, world_size, n_epochs):
    """Load metrics from workdir JSON files
    
    Args:
        workdir: Path to the workdir containing result JSON files
        world_size: Number of devices/ranks
        n_epochs: Number of epochs
    
    Returns:
        Dict mapping rank to list of epoch metrics
    """
    workdir_path = Path(workdir)
    metrics_by_rank = {}
    
    for rank in range(world_size):
        metrics_by_rank[rank] = []
        for epoch_idx in range(n_epochs):
            filename = workdir_path / f"rank{rank}_results_epoch{epoch_idx}.json"
            if filename.exists():
                with open(filename, 'r') as f:
                    data = json.load(f)
                    metrics_by_rank[rank].append(data)
            else:
                print(f"Warning: {filename} not found")
    
    return metrics_by_rank


def plot_comparison(means, stds, labels, ylabel, title, filename, output_dir='submit/figures'):
    """Create a bar plot comparing metrics"""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(means))
    
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'][:len(means)]
    
    ax.bar(x, means, yerr=stds,
           align='center', alpha=0.7, ecolor='red', capsize=10, width=0.6,
           color=colors)
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean, f'{mean:.2f}\n±{std:.2f}', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_file}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze Problem 1.3 metrics')
    parser.add_argument('--workdir', type=str, default='./workdir',
                        help='Path to workdir containing result JSON files')
    parser.add_argument('--single-world-size', type=int, default=1,
                        help='World size for single device run')
    parser.add_argument('--multi-world-size', type=int, default=2,
                        help='World size for multi-device run')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--output-dir', type=str, default='submit/figures',
                        help='Output directory for figures')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Problem 1.3: Performance Comparison Analysis")
    print("="*60)
    
    # Load single device metrics
    print(f"\nLoading single device metrics (world_size={args.single_world_size})...")
    metrics_single = load_metrics_from_workdir(
        args.workdir, args.single_world_size, args.n_epochs
    )
    
    if not metrics_single or not metrics_single.get(0):
        print("Error: No metrics found for single device run!")
        print(f"Please ensure workdir '{args.workdir}' contains rank0_results_epoch*.json files")
        return
    
    aggregated_single = compute_aggregated_metrics(metrics_single, drop_first_epoch=True)
    
    print(f"Single Device Metrics (excluding first epoch):")
    print(f"  Training Time: {aggregated_single['avg_training_time']:.2f} ± {aggregated_single['std_training_time']:.2f} seconds")
    print(f"  Tokens/Second: {aggregated_single['avg_tokens_per_sec']:.2f} ± {aggregated_single['std_tokens_per_sec']:.2f}")
    
    # Load multi-device metrics
    print(f"\nLoading multi-device metrics (world_size={args.multi_world_size})...")
    metrics_multi = load_metrics_from_workdir(
        args.workdir, args.multi_world_size, args.n_epochs
    )
    
    if not metrics_multi or not metrics_multi.get(0):
        print("Error: No metrics found for multi-device run!")
        print(f"Please ensure workdir '{args.workdir}' contains rank*_results_epoch*.json files")
        return
    
    aggregated_multi = compute_aggregated_metrics(metrics_multi, drop_first_epoch=True)
    
    print(f"Multi-Device Metrics (excluding first epoch):")
    print(f"  Training Time: {aggregated_multi['avg_training_time']:.2f} ± {aggregated_multi['std_training_time']:.2f} seconds")
    print(f"  Tokens/Second: {aggregated_multi['avg_tokens_per_sec']:.2f} ± {aggregated_multi['std_tokens_per_sec']:.2f}")
    
    # Calculate speedup (with validation to avoid division by zero)
    if aggregated_multi['avg_training_time'] > 0 and aggregated_single['avg_tokens_per_sec'] > 0:
        time_speedup = aggregated_single['avg_training_time'] / aggregated_multi['avg_training_time']
        throughput_speedup = aggregated_multi['avg_tokens_per_sec'] / aggregated_single['avg_tokens_per_sec']
        
        print(f"\nSpeedup:")
        print(f"  Training Time Speedup: {time_speedup:.2f}x")
        print(f"  Throughput Speedup: {throughput_speedup:.2f}x")
    else:
        print(f"\nSpeedup: Cannot compute (insufficient data)")
    
    # Create visualizations
    print(f"\nGenerating plots...")
    
    # Plot Training Time comparison
    plot_comparison(
        means=[aggregated_single['avg_training_time'], aggregated_multi['avg_training_time']],
        stds=[aggregated_single['std_training_time'], aggregated_multi['std_training_time']],
        labels=[f'Single Device\\n(world_size={args.single_world_size})', 
                f'Multi-Device\\n(world_size={args.multi_world_size})'],
        ylabel='Training Time (seconds)',
        title='Training Time Comparison',
        filename='training_time_comparison.png',
        output_dir=args.output_dir
    )
    
    # Plot Tokens Per Second comparison
    plot_comparison(
        means=[aggregated_single['avg_tokens_per_sec'], aggregated_multi['avg_tokens_per_sec']],
        stds=[aggregated_single['std_tokens_per_sec'], aggregated_multi['std_tokens_per_sec']],
        labels=[f'Single Device\\n(world_size={args.single_world_size})', 
                f'Multi-Device\\n(world_size={args.multi_world_size})'],
        ylabel='Tokens Per Second',
        title='Throughput Comparison',
        filename='tokens_per_second_comparison.png',
        output_dir=args.output_dir
    )
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Figures saved to: {args.output_dir}/")
    print("  - training_time_comparison.png")
    print("  - tokens_per_second_comparison.png")
    print("="*60)


if __name__ == "__main__":
    main()
