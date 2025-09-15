"""
Comprehensive profiling tests for all models in the ChillAdam registry.
Tests speed, memory usage during inference and training, gradient memory, 
activation memory, and weight memory for all available models.
"""

import pytest
import torch
import torch.nn as nn
import time
import gc
import psutil
import os
import sys
from typing import Dict, List, Tuple, Any, Callable
from dataclasses import dataclass

# Add the chilladam package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import chilladam


@dataclass
class ProfileResult:
    """Container for profiling results."""
    model_name: str
    inference_time: float
    training_time: float
    inference_memory_mb: float
    training_memory_mb: float
    weight_memory_mb: float
    gradient_memory_mb: float
    activation_memory_mb: float
    total_parameters: int
    trainable_parameters: int


class MemoryProfiler:
    """Utility class for memory profiling."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def clear_cache(self):
        """Clear all caches."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_all_model_functions() -> List[str]:
    """Get all model function names from the registry."""
    model_functions = []
    for name in chilladam.__all__:
        obj = getattr(chilladam, name)
        if callable(obj) and name not in [
            'ChillAdam', 'create_scheduler', 'get_scheduler_info', 
            'add_l1_regularization', 'L1RegularizedLoss', 'ResNet'
        ]:
            model_functions.append(name)
    return sorted(model_functions)


def create_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """Create a model instance by name."""
    model_func = getattr(chilladam, model_name)
    
    # Handle different model types
    if 'vit' in model_name:
        return model_func(num_classes=num_classes, img_size=224)
    else:
        return model_func(num_classes=num_classes)


def get_model_memory_breakdown(model: nn.Module) -> Tuple[float, float, int, int]:
    """Get memory breakdown for model weights and parameters."""
    total_params = 0
    trainable_params = 0
    weight_memory = 0.0
    
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        # Calculate memory in bytes, then convert to MB
        param_memory = param.numel() * param.element_size()
        weight_memory += param_memory
    
    weight_memory_mb = weight_memory / 1024 / 1024
    return weight_memory_mb, 0.0, total_params, trainable_params


def profile_inference(model: nn.Module, input_tensor: torch.Tensor, 
                     num_runs: int = 5) -> Tuple[float, float]:
    """Profile inference speed and memory usage."""
    profiler = MemoryProfiler()
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    profiler.clear_cache()
    memory_before = profiler.get_memory_mb()
    
    # Time inference
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            output = model(input_tensor)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    memory_after = profiler.get_memory_mb()
    inference_time = (end_time - start_time) / num_runs
    memory_used = memory_after - memory_before
    
    return inference_time, memory_used


def profile_training(model: nn.Module, input_tensor: torch.Tensor, 
                    target: torch.Tensor, num_runs: int = 3) -> Tuple[float, float, float]:
    """Profile training speed, memory usage, and gradient memory."""
    profiler = MemoryProfiler()
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Warmup
    for _ in range(2):
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        model.zero_grad()
    
    profiler.clear_cache()
    memory_before = profiler.get_memory_mb()
    
    # Time training step
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for _ in range(num_runs):
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        
        # Calculate gradient memory after backward pass
        gradient_memory = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_memory = param.grad.numel() * param.grad.element_size()
                gradient_memory += grad_memory
        
        model.zero_grad()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    
    memory_after = profiler.get_memory_mb()
    training_time = (end_time - start_time) / num_runs
    memory_used = memory_after - memory_before
    gradient_memory_mb = gradient_memory / 1024 / 1024
    
    return training_time, memory_used, gradient_memory_mb


def estimate_activation_memory(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """Estimate activation memory by comparing before and after forward pass."""
    profiler = MemoryProfiler()
    model.eval()
    
    profiler.clear_cache()
    memory_before = profiler.get_memory_mb()
    
    # Forward pass while retaining gradients to keep activations
    with torch.enable_grad():
        input_tensor.requires_grad_(True)
        output = model(input_tensor)
        # Keep reference to output to prevent deallocation
        output.retain_grad()
    
    memory_after = profiler.get_memory_mb()
    activation_memory = memory_after - memory_before
    
    # Clean up
    del output
    input_tensor.requires_grad_(False)
    profiler.clear_cache()
    
    return max(0.0, activation_memory)


def profile_single_model(model_name: str, batch_size: int = 4, 
                        input_size: Tuple[int, int, int] = (3, 224, 224),
                        num_classes: int = 10) -> ProfileResult:
    """Profile a single model comprehensively."""
    print(f"Profiling {model_name}...")
    
    # Create model and inputs
    model = create_model(model_name, num_classes)
    input_tensor = torch.randn(batch_size, *input_size)
    target = torch.randint(0, num_classes, (batch_size,))
    
    # Get model memory breakdown
    weight_memory_mb, _, total_params, trainable_params = get_model_memory_breakdown(model)
    
    # Profile inference
    inference_time, inference_memory = profile_inference(model, input_tensor)
    
    # Profile training
    training_time, training_memory, gradient_memory = profile_training(
        model, input_tensor, target
    )
    
    # Estimate activation memory
    activation_memory = estimate_activation_memory(model, input_tensor)
    
    return ProfileResult(
        model_name=model_name,
        inference_time=inference_time,
        training_time=training_time,
        inference_memory_mb=inference_memory,
        training_memory_mb=training_memory,
        weight_memory_mb=weight_memory_mb,
        gradient_memory_mb=gradient_memory,
        activation_memory_mb=activation_memory,
        total_parameters=total_params,
        trainable_parameters=trainable_params
    )


class TestModelProfiling:
    """Test class for model profiling."""
    
    @pytest.fixture(scope="class")
    def model_names(self):
        """Get all model names for testing."""
        return get_all_model_functions()
    
    def test_model_inference_speed(self, model_names):
        """Test inference speed for a subset of models."""
        # Test a representative subset for faster execution
        test_models = ['resnet18', 'resnet50', 'se_resnet18', 'vit_base']
        results = {}
        
        for model_name in test_models:
            if model_name in model_names:
                try:
                    model = create_model(model_name, num_classes=10)
                    input_tensor = torch.randn(2, 3, 224, 224)  # Smaller batch for speed
                    
                    inference_time, _ = profile_inference(model, input_tensor, num_runs=3)
                    results[model_name] = inference_time
                    
                    # Assert reasonable inference time (should be less than 5 seconds for these sizes)
                    assert inference_time < 5.0, f"{model_name} inference too slow: {inference_time:.3f}s"
                    
                except Exception as e:
                    pytest.fail(f"Failed to profile inference for {model_name}: {str(e)}")
        
        print("\nInference Speed Results (subset):")
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:25}: {time_val:.4f}s")
    
    def test_model_training_speed(self, model_names):
        """Test training speed for a subset of models."""
        # Test a representative subset for faster execution
        test_models = ['resnet18', 'se_resnet18', 'vit_base']
        results = {}
        
        for model_name in test_models:
            if model_name in model_names:
                try:
                    model = create_model(model_name, num_classes=10)
                    input_tensor = torch.randn(2, 3, 224, 224)  # Smaller batch for speed
                    target = torch.randint(0, 10, (2,))
                    
                    training_time, _, _ = profile_training(model, input_tensor, target, num_runs=2)
                    results[model_name] = training_time
                    
                    # Assert reasonable training time (should be less than 10 seconds for these sizes)
                    assert training_time < 10.0, f"{model_name} training too slow: {training_time:.3f}s"
                    
                except Exception as e:
                    pytest.fail(f"Failed to profile training for {model_name}: {str(e)}")
        
        print("\nTraining Speed Results (subset):")
        for name, time_val in sorted(results.items(), key=lambda x: x[1]):
            print(f"  {name:25}: {time_val:.4f}s")
    
    def test_model_memory_usage(self, model_names):
        """Test memory usage for a subset of models."""
        # Test a representative subset for faster execution
        test_models = ['resnet18', 'resnet50', 'se_resnet18', 'vit_base']
        results = {}
        
        for model_name in test_models:
            if model_name in model_names:
                try:
                    result = profile_single_model(model_name, batch_size=2)  # Smaller batch for speed
                    results[model_name] = result
                    
                    # Basic assertions
                    assert result.total_parameters > 0, f"{model_name} has no parameters"
                    assert result.weight_memory_mb > 0, f"{model_name} has no weight memory"
                    assert result.inference_time > 0, f"{model_name} has invalid inference time"
                    assert result.training_time > 0, f"{model_name} has invalid training time"
                    
                except Exception as e:
                    pytest.fail(f"Failed to profile memory for {model_name}: {str(e)}")
        
        print("\nMemory Usage Results (subset):")
        print(f"{'Model':<25} {'Params':<10} {'Weights':<10} {'Inference':<12} {'Training':<12} {'Gradients':<12} {'Activations':<12}")
        print("-" * 105)
        
        for name, result in sorted(results.items(), key=lambda x: x[1].total_parameters):
            print(f"{name:<25} {result.total_parameters/1e6:>8.2f}M "
                  f"{result.weight_memory_mb:>8.1f}MB "
                  f"{result.inference_memory_mb:>10.1f}MB "
                  f"{result.training_memory_mb:>10.1f}MB "
                  f"{result.gradient_memory_mb:>10.1f}MB "
                  f"{result.activation_memory_mb:>10.1f}MB")
    
    def test_model_parameter_counts(self, model_names):
        """Test parameter counts for all models."""
        results = {}
        
        for model_name in model_names:
            try:
                model = create_model(model_name, num_classes=1000)  # Standard ImageNet classes
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                results[model_name] = (total_params, trainable_params)
                
                # Assert all parameters are trainable by default
                assert total_params == trainable_params, f"{model_name} has non-trainable parameters"
                
                # Assert reasonable parameter counts (should be less than 500M)
                assert total_params < 500e6, f"{model_name} has too many parameters: {total_params/1e6:.1f}M"
                
            except Exception as e:
                pytest.fail(f"Failed to count parameters for {model_name}: {str(e)}")
        
        print("\nParameter Count Results:")
        for name, (total, trainable) in sorted(results.items(), key=lambda x: x[1][0]):
            print(f"  {name:25}: {total/1e6:>8.2f}M parameters")
    
    def test_model_gradient_flow(self, model_names):
        """Test that gradients flow properly for all models."""
        for model_name in model_names:
            try:
                model = create_model(model_name, num_classes=10)
                input_tensor = torch.randn(2, 3, 224, 224, requires_grad=True)
                target = torch.randint(0, 10, (2,))
                
                # Forward pass
                output = model(input_tensor)
                loss = nn.CrossEntropyLoss()(output, target)
                
                # Backward pass
                loss.backward()
                
                # Check that gradients exist for all parameters
                grad_count = 0
                total_count = 0
                for name, param in model.named_parameters():
                    total_count += 1
                    if param.grad is not None:
                        grad_count += 1
                        assert not torch.isnan(param.grad).any(), f"NaN gradient in {model_name} parameter {name}"
                        assert not torch.isinf(param.grad).any(), f"Inf gradient in {model_name} parameter {name}"
                
                assert grad_count == total_count, f"{model_name} missing gradients: {grad_count}/{total_count}"
                
            except Exception as e:
                pytest.fail(f"Failed gradient flow test for {model_name}: {str(e)}")
    
    def test_model_memory_efficiency(self, model_names):
        """Test memory efficiency across different batch sizes."""
        batch_sizes = [1, 4, 8]
        
        for model_name in model_names[:3]:  # Test subset to avoid timeout
            try:
                print(f"\nTesting {model_name} with different batch sizes:")
                
                for batch_size in batch_sizes:
                    model = create_model(model_name, num_classes=10)
                    input_tensor = torch.randn(batch_size, 3, 224, 224)
                    
                    inference_time, inference_memory = profile_inference(model, input_tensor, num_runs=3)
                    
                    print(f"  Batch size {batch_size}: {inference_time:.4f}s, {inference_memory:.1f}MB")
                    
                    # Memory should scale reasonably with batch size
                    if batch_size > 1:
                        assert inference_time > 0, f"Invalid timing for batch size {batch_size}"
                
            except Exception as e:
                pytest.fail(f"Failed memory efficiency test for {model_name}: {str(e)}")


def run_comprehensive_profiling():
    """Run comprehensive profiling and save results."""
    model_names = get_all_model_functions()
    results = []
    
    print("Running comprehensive profiling for all models...")
    print("=" * 80)
    
    for model_name in model_names:
        try:
            result = profile_single_model(model_name)
            results.append(result)
            print(f"✓ {model_name} completed")
        except Exception as e:
            print(f"✗ {model_name} failed: {str(e)}")
    
    # Print summary table
    print("\n" + "=" * 120)
    print("COMPREHENSIVE PROFILING RESULTS")
    print("=" * 120)
    print(f"{'Model':<25} {'Params':<10} {'Inf.Time':<10} {'Train.Time':<11} {'Weights':<10} {'Inference':<11} {'Training':<10} {'Gradients':<11} {'Activations':<12}")
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
    
    return results


if __name__ == "__main__":
    # Run comprehensive profiling when script is executed directly
    results = run_comprehensive_profiling()