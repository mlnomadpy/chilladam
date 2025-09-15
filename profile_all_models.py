#!/usr/bin/env python3
"""
Comprehensive model profiling script for all models in ChillAdam registry.
Run this script to get detailed profiling results for all models.

Usage:
    python profile_all_models.py [--batch-size 4] [--output results.csv]
"""

import argparse
import csv
import sys
import os
from pathlib import Path

# Add the chilladam package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chilladam.test.test_model_profiling import (
    get_all_model_functions, profile_single_model, ProfileResult
)


def save_results_to_csv(results: list, filename: str):
    """Save profiling results to CSV file."""
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'model_name', 'total_parameters', 'trainable_parameters',
            'inference_time_s', 'training_time_s', 'weight_memory_mb',
            'inference_memory_mb', 'training_memory_mb', 'gradient_memory_mb',
            'activation_memory_mb'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow({
                'model_name': result.model_name,
                'total_parameters': result.total_parameters,
                'trainable_parameters': result.trainable_parameters,
                'inference_time_s': result.inference_time,
                'training_time_s': result.training_time,
                'weight_memory_mb': result.weight_memory_mb,
                'inference_memory_mb': result.inference_memory_mb,
                'training_memory_mb': result.training_memory_mb,
                'gradient_memory_mb': result.gradient_memory_mb,
                'activation_memory_mb': result.activation_memory_mb,
            })


def main():
    parser = argparse.ArgumentParser(description='Profile all ChillAdam models')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for profiling (default: 4)')
    parser.add_argument('--output', type=str, default='model_profiling_results.csv',
                        help='Output CSV file (default: model_profiling_results.csv)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to profile (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick profiling mode (subset of models)')
    
    args = parser.parse_args()
    
    # Get model names
    all_models = get_all_model_functions()
    
    if args.quick:
        # Quick mode: test representative models
        models_to_test = ['resnet18', 'resnet50', 'se_resnet18', 'se_resnet50', 
                         'yat_resnet18', 'yat_se_resnet18', 'vit_base', 'vit_large']
        models_to_test = [m for m in models_to_test if m in all_models]
    elif args.models:
        models_to_test = [m for m in args.models if m in all_models]
        if not models_to_test:
            print(f"Error: No valid models found in {args.models}")
            print(f"Available models: {', '.join(all_models)}")
            return 1
    else:
        models_to_test = all_models
    
    print(f"Profiling {len(models_to_test)} models with batch size {args.batch_size}")
    print("=" * 80)
    
    results = []
    failed_models = []
    
    for i, model_name in enumerate(models_to_test, 1):
        try:
            print(f"[{i:2d}/{len(models_to_test):2d}] Profiling {model_name}...")
            result = profile_single_model(model_name, batch_size=args.batch_size)
            results.append(result)
            print(f"✓ {model_name} completed "
                  f"({result.total_parameters/1e6:.1f}M params, "
                  f"{result.inference_time:.3f}s inf, "
                  f"{result.training_time:.3f}s train)")
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
            failed_models.append((model_name, str(e)))
    
    # Print comprehensive results
    print("\n" + "=" * 120)
    print("COMPREHENSIVE PROFILING RESULTS")
    print("=" * 120)
    print(f"{'Model':<25} {'Params':<10} {'Inf.Time':<10} {'Train.Time':<11} "
          f"{'Weights':<10} {'Inference':<11} {'Training':<10} {'Gradients':<11} {'Activations':<12}")
    print("-" * 120)
    
    for result in sorted(results, key=lambda x: x.total_parameters):
        print(f"{result.model_name:<25} "
              f"{result.total_parameters/1e6:>8.2f}M "
              f"{result.inference_time:>8.4f}s "
              f"{result.training_time:>9.4f}s "
              f"{result.weight_memory_mb:>8.1f}MB "
              f"{result.inference_memory_mb:>9.1f}MB "
              f"{result.training_memory_mb:>8.1f}MB "
              f"{result.gradient_memory_mb:>9.1f}MB "
              f"{result.activation_memory_mb:>10.1f}MB")
    
    # Print summary statistics
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        total_params = [r.total_parameters for r in results]
        inference_times = [r.inference_time for r in results]
        training_times = [r.training_time for r in results]
        weight_memory = [r.weight_memory_mb for r in results]
        
        print(f"Models profiled: {len(results)}")
        print(f"Parameter range: {min(total_params)/1e6:.1f}M - {max(total_params)/1e6:.1f}M")
        print(f"Inference time range: {min(inference_times):.3f}s - {max(inference_times):.3f}s")
        print(f"Training time range: {min(training_times):.3f}s - {max(training_times):.3f}s")
        print(f"Weight memory range: {min(weight_memory):.1f}MB - {max(weight_memory):.1f}MB")
    
    # Print failed models
    if failed_models:
        print(f"\n⚠️  {len(failed_models)} models failed:")
        for model_name, error in failed_models:
            print(f"  - {model_name}: {error}")
    
    # Save results to CSV
    if results:
        save_results_to_csv(results, args.output)
        print(f"\n💾 Results saved to {args.output}")
    
    print(f"\n✅ Profiling completed: {len(results)} successful, {len(failed_models)} failed")
    return 0 if not failed_models else 1


if __name__ == '__main__':
    sys.exit(main())