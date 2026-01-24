"""
Main Demo Script for Multi-Stage Adaptive Inference Pipeline
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from pipeline.adaptive_pipeline import AdaptivePipeline
from utils.metrics import MetricsTracker
from utils.visualization import create_summary_dashboard


def create_sample_video(output_path: str, duration: int = 5, fps: int = 30):
    """
    Create a sample video for testing
    
    Args:
        output_path: Path to save video
        duration: Duration in seconds
        fps: Frames per second
    """
    try:
        import cv2
        
        # Video properties
        width, height = 640, 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        total_frames = duration * fps
        
        for i in range(total_frames):
            # Create a frame with random noise
            frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            # Add some text
            text = f"Sample Video - Frame {i+1}/{total_frames}"
            cv2.putText(frame, text, (50, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        print(f"✓ Created sample video: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠ Could not create sample video: {e}")
        return False


def run_demo_mode():
    """Run demo with sample videos"""
    print("\n" + "="*70)
    print("DEMO MODE: Multi-Stage Adaptive Inference Pipeline")
    print("="*70)
    
    # Create sample videos directory
    sample_dir = os.path.join(project_root, "sample_videos")
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create sample videos
    print("\nCreating sample videos...")
    sample_videos = []
    
    for i in range(5):
        video_path = os.path.join(sample_dir, f"sample_video_{i+1}.mp4")
        if create_sample_video(video_path, duration=3, fps=30):
            sample_videos.append(video_path)
    
    if not sample_videos:
        print("\n⚠ Could not create sample videos.")
        print("Please provide your own video files using: python demo.py --video path/to/video.mp4")
        return
    
    # Initialize pipeline
    print("\n" + "="*70)
    print("Initializing Adaptive Pipeline...")
    print("="*70)
    
    pipeline = AdaptivePipeline()
    tracker = MetricsTracker()
    
    # Process videos
    print("\n" + "="*70)
    print("Processing Sample Videos...")
    print("="*70)
    
    results = []
    
    for video_path in sample_videos:
        result = pipeline.predict(video_path)
        results.append(result)
        tracker.add_result(result)
    
    # Print statistics
    print("\n" + "="*70)
    print("PIPELINE STATISTICS")
    print("="*70)
    pipeline.print_statistics()
    
    # Print metrics
    tracker.print_summary()
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70)
    
    results_dir = os.path.join(project_root, "results")
    create_summary_dashboard(results, results_dir)
    
    # Save results
    results_file = os.path.join(results_dir, "demo_results.json")
    tracker.save_results(results_file)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {results_dir}/")
    print("\nCheck the following files:")
    print("  - demo_results.json (detailed results)")
    print("  - exit_distribution.png")
    print("  - time_comparison.png")
    print("  - confidence_distribution.png")
    print("  - pipeline_flow.png")
    print("\n" + "="*70)


def run_single_video(video_path: str):
    """
    Run pipeline on a single video
    
    Args:
        video_path: Path to video file
    """
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        return
    
    print("\n" + "="*70)
    print("Single Video Analysis")
    print("="*70)
    
    # Initialize pipeline
    pipeline = AdaptivePipeline()
    
    # Process video
    result = pipeline.predict(video_path)
    
    # Print result
    print("\n" + "="*70)
    print("RESULT SUMMARY")
    print("="*70)
    print(f"Video:       {os.path.basename(video_path)}")
    print(f"Prediction:  {result['label']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Confidence:  {result['confidence']:.4f}")
    print(f"Exit Stage:  {result['exit_stage']}")
    print(f"Time:        {result['total_time']:.2f}s")
    print("="*70 + "\n")


def run_batch_mode(video_dir: str):
    """
    Run pipeline on all videos in a directory
    
    Args:
        video_dir: Directory containing videos
    """
    if not os.path.exists(video_dir):
        print(f"Error: Directory not found: {video_dir}")
        return
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(Path(video_dir).glob(f"*{ext}"))
    
    if not video_files:
        print(f"No video files found in: {video_dir}")
        return
    
    print(f"\nFound {len(video_files)} video(s)")
    
    # Initialize pipeline and tracker
    pipeline = AdaptivePipeline()
    tracker = MetricsTracker()
    
    # Process videos
    results = []
    for video_path in video_files:
        result = pipeline.predict(str(video_path))
        results.append(result)
        tracker.add_result(result)
    
    # Print statistics
    pipeline.print_statistics()
    tracker.print_summary()
    
    # Create visualizations
    results_dir = os.path.join(project_root, "results")
    create_summary_dashboard(results, results_dir)
    
    # Save results
    results_file = os.path.join(results_dir, "batch_results.json")
    tracker.save_results(results_file)
    
    print(f"\n✓ Results saved to: {results_dir}/")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Multi-Stage Adaptive Inference Pipeline for Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo with sample videos
  python demo.py --demo
  
  # Analyze single video
  python demo.py --video path/to/video.mp4
  
  # Analyze all videos in directory
  python demo.py --batch path/to/video/directory
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run demo mode with sample videos')
    parser.add_argument('--video', type=str,
                       help='Path to single video file')
    parser.add_argument('--batch', type=str,
                       help='Path to directory containing videos')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "#"*70)
    print("# Multi-Stage Adaptive Inference Pipeline")
    print("# Deepfake Video Detection Demo")
    print("#"*70)
    
    # Run appropriate mode
    if args.demo:
        run_demo_mode()
    elif args.video:
        run_single_video(args.video)
    elif args.batch:
        run_batch_mode(args.batch)
    else:
        # Default to demo mode
        print("\nNo arguments provided. Running in demo mode...")
        print("Use --help to see all options.\n")
        run_demo_mode()


if __name__ == "__main__":
    main()
