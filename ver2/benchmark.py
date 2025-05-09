import os
import torch
import argparse
import time
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import SimCLRClassifier
from dataset import get_dataloaders
from utils import evaluate_model, visualize_embeddings

def benchmark_inference_time(model, data_loader, device, num_runs=100):
    """
    Benchmark inference time
    """
    model.eval()
    
    # Warm-up runs
    for images, _ in data_loader:
        images = images.to(device)
        _ = model(images)
        break
    
    # Time inference
    batch_times = []
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            
            # Run inference multiple times to get average
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(images)
                torch.cuda.synchronize() if device.type == 'cuda' else None
                end_time = time.time()
                
                batch_times.append(end_time - start_time)
                
            break  # Only use the first batch for timing
    
    # Calculate statistics
    batch_size = images.shape[0]
    latency_mean = np.mean(batch_times) * 1000  # Convert to ms
    latency_std = np.std(batch_times) * 1000
    throughput = batch_size / np.mean(batch_times)
    
    print(f"Inference Benchmark:")
    print(f"  Batch size: {batch_size}")
    print(f"  Latency: {latency_mean:.2f} ± {latency_std:.2f} ms")
    print(f"  Throughput: {throughput:.2f} images/sec")
    
    return {
        "batch_size": batch_size,
        "latency_ms": latency_mean,
        "latency_std_ms": latency_std,
        "throughput": throughput
    }

def main():
    parser = argparse.ArgumentParser(description='Benchmark SimCLR model')
    parser.add_argument('--data-dir', type=str, required=True, help='path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--model-path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--visualize', action='store_true', help='visualize embeddings')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = SimCLRClassifier(num_classes=5).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")
    
    # Get data loader
    _, val_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Run benchmark
    benchmark_results = benchmark_inference_time(model, val_loader, device)
    
    # Evaluate model
    print("\nEvaluating model...")
    report = evaluate_model(model, val_loader, device)
    print("\nClassification Report:")
    print(report)
    
    # Visualize embeddings
    if args.visualize:
        print("\nVisualizing embeddings...")
        visualize_embeddings(model, val_loader, device)
        print("Embeddings visualization saved to tsne_embeddings.png")
    
    # Save benchmark results
    if not os.path.exists('results'):
        os.makedirs('results')
        
    np.save('results/benchmark_results.npy', benchmark_results)
    
    print("\nBenchmark complete!")

if __name__ == '__main__':
    main() 