"""
Visualization utilities for pipeline results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict
import os


def plot_exit_distribution(results: List[Dict], save_path: str = None):
    """
    Plot distribution of videos across stages
    
    Args:
        results: List of prediction results
        save_path: Path to save figure
    """
    stage_counts = {1: 0, 2: 0, 3: 0}
    
    for result in results:
        stage = result.get('exit_stage', 3)
        stage_counts[stage] += 1
    
    stages = ['Stage 1\n(Fast)', 'Stage 2\n(Balanced)', 'Stage 3\n(Accurate)']
    counts = [stage_counts[1], stage_counts[2], stage_counts[3]]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(stages, counts, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title('Exit Stage Distribution\nMulti-Stage Adaptive Pipeline', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_time_comparison(results: List[Dict], save_path: str = None):
    """
    Plot time comparison between adaptive and full processing
    
    Args:
        results: List of prediction results
        save_path: Path to save figure
    """
    # Calculate times
    stage_times = {1: [], 2: [], 3: []}
    
    for result in results:
        stage = result.get('exit_stage', 3)
        time = result.get('total_time', 0)
        stage_times[stage].append(time)
    
    # Calculate averages
    avg_times = {
        1: np.mean(stage_times[1]) if stage_times[1] else 0,
        2: np.mean(stage_times[2]) if stage_times[2] else 0,
        3: np.mean(stage_times[3]) if stage_times[3] else 0
    }
    
    # Estimate full processing time (assume Stage 3 time for all)
    adaptive_avg = sum(r.get('total_time', 0) for r in results) / len(results) if results else 0
    full_avg = avg_times[3] if avg_times[3] > 0 else adaptive_avg * 2
    
    methods = ['Adaptive\nPipeline', 'Full Processing\n(Always Stage 3)']
    times = [adaptive_avg, full_avg]
    colors = ['#3498db', '#95a5a6']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, times, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add savings percentage
    savings = ((full_avg - adaptive_avg) / full_avg * 100) if full_avg > 0 else 0
    ax.text(0.5, max(times) * 0.9, f'Time Savings: {savings:.1f}%',
            ha='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.set_ylabel('Average Time per Video (seconds)', fontsize=12)
    ax.set_title('Processing Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confidence_distribution(results: List[Dict], save_path: str = None):
    """
    Plot confidence score distribution
    
    Args:
        results: List of prediction results
        save_path: Path to save figure
    """
    confidences = [r.get('confidence', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(confidences, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    
    # Add mean line
    mean_conf = np.mean(confidences) if confidences else 0
    ax.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
    
    ax.set_xlabel('Confidence Score', fontsize=12)
    ax.set_ylabel('Number of Videos', fontsize=12)
    ax.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_pipeline_flow(results: List[Dict], save_path: str = None):
    """
    Create a Sankey-style flow diagram showing video progression through stages
    
    Args:
        results: List of prediction results
        save_path: Path to save figure
    """
    total = len(results)
    stage1_exit = sum(1 for r in results if r.get('exit_stage') == 1)
    stage2_exit = sum(1 for r in results if r.get('exit_stage') == 2)
    stage3_exit = sum(1 for r in results if r.get('exit_stage') == 3)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define positions
    stages = [
        {'name': 'Input\nVideos', 'x': 0, 'y': 0.5, 'count': total},
        {'name': 'Stage 1\nFast', 'x': 1, 'y': 0.5, 'count': total},
        {'name': 'Stage 2\nBalanced', 'x': 2, 'y': 0.5, 'count': total - stage1_exit},
        {'name': 'Stage 3\nAccurate', 'x': 3, 'y': 0.5, 'count': total - stage1_exit - stage2_exit},
    ]
    
    exits = [
        {'name': f'Exit\n{stage1_exit} videos', 'x': 1, 'y': 0.2, 'count': stage1_exit},
        {'name': f'Exit\n{stage2_exit} videos', 'x': 2, 'y': 0.2, 'count': stage2_exit},
        {'name': f'Exit\n{stage3_exit} videos', 'x': 3, 'y': 0.2, 'count': stage3_exit},
    ]
    
    # Draw stage boxes
    for stage in stages:
        width = 0.15
        height = 0.15
        rect = plt.Rectangle((stage['x'] - width/2, stage['y'] - height/2),
                            width, height, facecolor='lightblue',
                            edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(stage['x'], stage['y'], f"{stage['name']}\n({stage['count']})",
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Draw exit boxes
    for exit_box in exits:
        if exit_box['count'] > 0:
            width = 0.15
            height = 0.1
            rect = plt.Rectangle((exit_box['x'] - width/2, exit_box['y'] - height/2),
                                width, height, facecolor='lightgreen',
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            ax.text(exit_box['x'], exit_box['y'], exit_box['name'],
                    ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw arrows
    arrow_props = dict(arrowstyle='->', lw=2, color='black')
    
    # Stage connections
    for i in range(len(stages) - 1):
        ax.annotate('', xy=(stages[i+1]['x'] - 0.08, stages[i+1]['y']),
                   xytext=(stages[i]['x'] + 0.08, stages[i]['y']),
                   arrowprops=arrow_props)
    
    # Exit arrows
    for i, exit_box in enumerate(exits):
        if exit_box['count'] > 0:
            ax.annotate('', xy=(exit_box['x'], exit_box['y'] + 0.05),
                       xytext=(stages[i+1]['x'], stages[i+1]['y'] - 0.08),
                       arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(0, 0.8)
    ax.axis('off')
    ax.set_title('Multi-Stage Pipeline Flow\nAdaptive Video Processing', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_summary_dashboard(results: List[Dict], save_dir: str = "results"):
    """
    Create a comprehensive dashboard with all visualizations
    
    Args:
        results: List of prediction results
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nGenerating visualization dashboard...")
    
    plot_exit_distribution(results, os.path.join(save_dir, "exit_distribution.png"))
    plot_time_comparison(results, os.path.join(save_dir, "time_comparison.png"))
    plot_confidence_distribution(results, os.path.join(save_dir, "confidence_distribution.png"))
    plot_pipeline_flow(results, os.path.join(save_dir, "pipeline_flow.png"))
    
    print(f"\n✓ Dashboard created in: {save_dir}/")
    print("  - exit_distribution.png")
    print("  - time_comparison.png")
    print("  - confidence_distribution.png")
    print("  - pipeline_flow.png")


if __name__ == "__main__":
    print("Visualization Module - Test Mode")
    print("="*60)
    
    # Create sample data
    sample_results = [
        {'exit_stage': 1, 'total_time': 1.2, 'confidence': 0.92},
        {'exit_stage': 1, 'total_time': 1.1, 'confidence': 0.88},
        {'exit_stage': 1, 'total_time': 1.3, 'confidence': 0.85},
        {'exit_stage': 2, 'total_time': 2.3, 'confidence': 0.65},
        {'exit_stage': 2, 'total_time': 2.1, 'confidence': 0.70},
        {'exit_stage': 3, 'total_time': 4.5, 'confidence': 0.55},
    ]
    
    print("\nGenerating sample visualizations...")
    create_summary_dashboard(sample_results, "test_results")
    
    print("\n✓ Module loaded successfully!")
