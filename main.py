# ==========================================
# main.py (ROOT LEVEL)
# ==========================================
#!/usr/bin/env python3
"""
CNN Architecture Comparison for 1D Signal Classification

Main entry point for comparing different CNN architectures on IASC dataset.
This script demonstrates implementations of methods from recent literature.

Usage:
    python main.py --model all           # Compare all 1D models
    python main.py --model park          # Evaluate Park 2D model  
    python main.py --model azimi         # Run specific model
    python main.py --help               # Show all options
"""

import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.compare_1d_models import compare_1d_models
from scripts.evaluate_park_model import evaluate_park_model
from models.model_utils import setup_gpu_memory


def main():
    """Main entry point with command line arguments."""
    parser = argparse.ArgumentParser(
        description="CNN Architecture Comparison for Signal Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --model all              # Compare all 1D models
  python main.py --model park             # Evaluate Park 2D model
  python main.py --model azimi            # Run only Azimi model
  python main.py --epochs 50 --folds 10   # Custom parameters
        """
    )
    
    parser.add_argument(
        '--model', 
        choices=['all', 'azimi', 'liu', 'rezende', 'park'],
        default='all',
        help='Which model(s) to evaluate (default: all)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of training epochs (default: 30)'
    )
    
    parser.add_argument(
        '--folds',
        type=int,
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size (default: 64)'
    )
    
    parser.add_argument(
        '--gpu-memory',
        type=int,
        default=7168,
        help='GPU memory limit in MB (default: 7168)'
    )
    
    args = parser.parse_args()
    
    # Setup GPU
    setup_gpu_memory(args.gpu_memory)
    
    print("="*60)
    print("CNN Architecture Comparison")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model(s): {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Folds: {args.folds}")
    print(f"  Batch size: {args.batch_size}")
    print("="*60)
    
    # Run selected models
    if args.model == 'all':
        print("\n Comparing all 1D CNN models...")
        compare_1d_models(
            epochs=args.epochs, 
            num_folds=args.folds, 
            batch_size=args.batch_size
        )
        
    elif args.model == 'park':
        print("\n Evaluating Park 2D CNN model...")
        evaluate_park_model(
            epochs=args.epochs,
            num_folds=args.folds,
            batch_size=args.batch_size
        )
        
    elif args.model in ['azimi', 'liu', 'rezende']:
        print(f"\n Evaluating {args.model.title()} model...")
        compare_1d_models(
            epochs=args.epochs,
            num_folds=args.folds,
            batch_size=args.batch_size,
            models=[args.model]  # Only run selected model
        )
    
    print("\n✅ Evaluation complete! Check the 'results/' folder for detailed outputs.")


if __name__ == '__main__':
    main()