"""
Performance Metrics and Tracking
"""

import time
import json
import os
from typing import Dict, List
from datetime import datetime


class MetricsTracker:
    """Track and analyze pipeline performance metrics"""
    
    def __init__(self):
        self.results = []
        self.start_time = None
        
    def add_result(self, result: Dict):
        """Add a prediction result"""
        self.results.append({
            **result,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_exit_distribution(self) -> Dict:
        """Get distribution of videos across stages"""
        if not self.results:
            return {}
        
        stage_counts = {1: 0, 2: 0, 3: 0}
        
        for result in self.results:
            stage = result.get('exit_stage', 3)
            stage_counts[stage] += 1
        
        total = len(self.results)
        
        return {
            'stage1': {
                'count': stage_counts[1],
                'percentage': (stage_counts[1] / total * 100) if total > 0 else 0
            },
            'stage2': {
                'count': stage_counts[2],
                'percentage': (stage_counts[2] / total * 100) if total > 0 else 0
            },
            'stage3': {
                'count': stage_counts[3],
                'percentage': (stage_counts[3] / total * 100) if total > 0 else 0
            }
        }
    
    def get_average_time(self) -> float:
        """Get average processing time"""
        if not self.results:
            return 0.0
        
        total_time = sum(r.get('total_time', 0) for r in self.results)
        return total_time / len(self.results)
    
    def get_time_savings(self, baseline_time: float = None) -> Dict:
        """
        Calculate time savings vs. always using Stage 3
        
        Args:
            baseline_time: Average time for Stage 3 processing
                          If None, estimates based on current data
        """
        if not self.results:
            return {}
        
        actual_time = sum(r.get('total_time', 0) for r in self.results)
        
        if baseline_time is None:
            # Estimate: assume Stage 3 takes 3x average Stage 1 time
            stage1_times = [r['total_time'] for r in self.results if r.get('exit_stage') == 1]
            if stage1_times:
                baseline_time = sum(stage1_times) / len(stage1_times) * 3
            else:
                baseline_time = actual_time  # No savings if no Stage 1 exits
        
        estimated_full_time = baseline_time * len(self.results)
        time_saved = estimated_full_time - actual_time
        savings_percentage = (time_saved / estimated_full_time * 100) if estimated_full_time > 0 else 0
        
        return {
            'actual_time': actual_time,
            'estimated_full_time': estimated_full_time,
            'time_saved': time_saved,
            'savings_percentage': savings_percentage
        }
    
    def get_label_distribution(self) -> Dict:
        """Get distribution of predictions"""
        if not self.results:
            return {}
        
        labels = [r.get('label', 'UNKNOWN') for r in self.results]
        
        deepfake_count = labels.count('DEEPFAKE')
        real_count = labels.count('REAL')
        total = len(labels)
        
        return {
            'deepfake': {
                'count': deepfake_count,
                'percentage': (deepfake_count / total * 100) if total > 0 else 0
            },
            'real': {
                'count': real_count,
                'percentage': (real_count / total * 100) if total > 0 else 0
            }
        }
    
    def get_confidence_stats(self) -> Dict:
        """Get confidence statistics"""
        if not self.results:
            return {}
        
        confidences = [r.get('confidence', 0) for r in self.results]
        
        return {
            'average': sum(confidences) / len(confidences) if confidences else 0,
            'min': min(confidences) if confidences else 0,
            'max': max(confidences) if confidences else 0
        }
    
    def print_summary(self):
        """Print comprehensive metrics summary"""
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*70)
        
        print(f"\nTotal Videos Processed: {len(self.results)}")
        
        # Exit distribution
        print("\n--- Exit Stage Distribution ---")
        exit_dist = self.get_exit_distribution()
        print(f"  Stage 1 (Fast):     {exit_dist['stage1']['count']:3d} videos ({exit_dist['stage1']['percentage']:5.1f}%)")
        print(f"  Stage 2 (Balanced): {exit_dist['stage2']['count']:3d} videos ({exit_dist['stage2']['percentage']:5.1f}%)")
        print(f"  Stage 3 (Accurate): {exit_dist['stage3']['count']:3d} videos ({exit_dist['stage3']['percentage']:5.1f}%)")
        
        # Label distribution
        print("\n--- Prediction Distribution ---")
        label_dist = self.get_label_distribution()
        print(f"  Deepfake: {label_dist['deepfake']['count']:3d} videos ({label_dist['deepfake']['percentage']:5.1f}%)")
        print(f"  Real:     {label_dist['real']['count']:3d} videos ({label_dist['real']['percentage']:5.1f}%)")
        
        # Time statistics
        print("\n--- Time Statistics ---")
        avg_time = self.get_average_time()
        print(f"  Average Time per Video: {avg_time:.2f}s")
        
        time_savings = self.get_time_savings()
        if time_savings:
            print(f"  Estimated Time Savings: {time_savings['savings_percentage']:.1f}%")
            print(f"  Time Saved: {time_savings['time_saved']:.2f}s")
        
        # Confidence statistics
        print("\n--- Confidence Statistics ---")
        conf_stats = self.get_confidence_stats()
        print(f"  Average Confidence: {conf_stats['average']:.3f}")
        print(f"  Min Confidence:     {conf_stats['min']:.3f}")
        print(f"  Max Confidence:     {conf_stats['max']:.3f}")
        
        print("="*70 + "\n")
    
    def save_results(self, filepath: str):
        """Save results to JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        data = {
            'summary': {
                'total_videos': len(self.results),
                'exit_distribution': self.get_exit_distribution(),
                'label_distribution': self.get_label_distribution(),
                'time_statistics': {
                    'average_time': self.get_average_time(),
                    'savings': self.get_time_savings()
                },
                'confidence_statistics': self.get_confidence_stats()
            },
            'results': self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Results saved to: {filepath}")
    
    def load_results(self, filepath: str):
        """Load results from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.results = data.get('results', [])
        print(f"✓ Loaded {len(self.results)} results from: {filepath}")


def compare_pipelines(adaptive_results: List[Dict], full_results: List[Dict]):
    """
    Compare adaptive pipeline vs. full processing
    
    Args:
        adaptive_results: Results from adaptive pipeline
        full_results: Results from always using Stage 3
    """
    print("\n" + "="*70)
    print("PIPELINE COMPARISON: Adaptive vs. Full Processing")
    print("="*70)
    
    adaptive_time = sum(r.get('total_time', 0) for r in adaptive_results)
    full_time = sum(r.get('total_time', 0) for r in full_results)
    
    time_saved = full_time - adaptive_time
    savings_pct = (time_saved / full_time * 100) if full_time > 0 else 0
    
    print(f"\nTotal Videos: {len(adaptive_results)}")
    print(f"\nAdaptive Pipeline:")
    print(f"  Total Time: {adaptive_time:.2f}s")
    print(f"  Avg Time:   {adaptive_time/len(adaptive_results):.2f}s")
    
    print(f"\nFull Processing (Always Stage 3):")
    print(f"  Total Time: {full_time:.2f}s")
    print(f"  Avg Time:   {full_time/len(full_results):.2f}s")
    
    print(f"\nSavings:")
    print(f"  Time Saved: {time_saved:.2f}s")
    print(f"  Percentage: {savings_pct:.1f}%")
    print(f"  Speedup:    {full_time/adaptive_time:.2f}x" if adaptive_time > 0 else "  N/A")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Metrics Tracker Module - Test Mode")
    print("="*60)
    
    # Create sample data
    tracker = MetricsTracker()
    
    # Simulate some results
    sample_results = [
        {'video': 'video1.mp4', 'label': 'DEEPFAKE', 'confidence': 0.92, 'exit_stage': 1, 'total_time': 1.2},
        {'video': 'video2.mp4', 'label': 'REAL', 'confidence': 0.88, 'exit_stage': 1, 'total_time': 1.1},
        {'video': 'video3.mp4', 'label': 'DEEPFAKE', 'confidence': 0.65, 'exit_stage': 2, 'total_time': 2.3},
        {'video': 'video4.mp4', 'label': 'REAL', 'confidence': 0.55, 'exit_stage': 3, 'total_time': 4.5},
    ]
    
    for result in sample_results:
        tracker.add_result(result)
    
    tracker.print_summary()
    
    print("✓ Module loaded successfully!")
