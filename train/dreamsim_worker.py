"""
DreamSim worker script for running in separate virtual environment.
This script receives batch image data via input file and outputs similarity scores via output file.
"""

import sys
import json
import base64
import io
import torch
from PIL import Image
from dreamsim import dreamsim
from torchvision import transforms
import os
import argparse
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import gc

def decode_base64_to_image(base64_str):
    """Decode base64 string to image bytes"""
    return base64.b64decode(base64_str.encode('utf-8'))


class ImagePairDataset(Dataset):
    """Dataset for image pairs with base64 encoded images"""
    
    def __init__(self, image_pairs):
        self.image_pairs = image_pairs
        self.img_size = 224
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        try:
            # Decode images and load images
            rendered_img_bytes = decode_base64_to_image(pair["rendered_img_base64"])
            ground_truth_img_bytes = decode_base64_to_image(pair["ground_truth_img_base64"])
            rendered_img = Image.open(io.BytesIO(rendered_img_bytes)).convert('RGB')
            ground_truth_img = Image.open(io.BytesIO(ground_truth_img_bytes)).convert('RGB')
            
            rendered_tensor = self.transform(rendered_img)
            ground_truth_tensor = self.transform(ground_truth_img)
            
            return {
                'index': idx,
                'rendered': rendered_tensor,
                'ground_truth': ground_truth_tensor,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing image pair {idx}: {e}", file=sys.stderr)
            # Return dummy tensors for failed pairs
            dummy_tensor = torch.zeros(3, self.img_size, self.img_size)
            return {
                'index': idx,
                'rendered': dummy_tensor,
                'ground_truth': dummy_tensor,
                'success': False,
                'error': str(e)
            }


def process_gpu_worker(gpu_id, image_pairs_chunk, process_id, total_processes, batch_size=32):
    """
    Process a subset of image pairs on specified GPU
    
    Args:
        gpu_id: GPU device ID
        image_pairs_chunk: List of image pairs to process
        process_id: Process identifier
        total_processes: Total number of processes
        batch_size: Batch size for processing
    """
    
    try:
        print(f"Process {process_id}/{total_processes} starting on GPU {gpu_id} with {len(image_pairs_chunk)} image pairs", file=sys.stderr)
        
        # Set CUDA device for this process
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Initialize DreamSim model in subprocess
        print(f"Process {process_id}: Initializing DreamSim model on GPU {gpu_id}...", file=sys.stderr)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Process {process_id}: Selected device: {device}", file=sys.stderr)
        model, _ = dreamsim(
            cache_dir=os.environ.get(
                "DREAMSIM_CACHE_DIR",
                os.path.expanduser("~/.cache/dreamsim"),
            ),
            device=device,
        )
        model.eval()
        print(f"Process {process_id}: DreamSim model loaded successfully on {device}", file=sys.stderr)
        
        # Create dataset and dataloader for this chunk
        dataset = ImagePairDataset(image_pairs_chunk)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,  # Use dynamic batch size
            shuffle=False, 
            num_workers=0,  # No additional workers in subprocess
            pin_memory=True if device.type == 'cuda' else False
        )
        
        results = []
        
        # Process images with periodic progress updates
        total_pairs = len(image_pairs_chunk)
        
        print(f"Process {process_id}: Starting batch similarity computation...", file=sys.stderr)
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Print progress at intervals
                if batch_idx % max(1, len(dataloader) // 5) == 0 or batch_idx == len(dataloader) - 1:
                    current_processed = min((batch_idx + 1) * batch_size, total_pairs)
                    progress_pct = current_processed / total_pairs * 100
                    print(f"Process {process_id}: Processing progress {current_processed}/{total_pairs} ({progress_pct:.1f}%)", file=sys.stderr)
                
                # Move batch to device
                rendered_batch = batch['rendered'].to(device)
                ground_truth_batch = batch['ground_truth'].to(device)
                indices = batch['index']
                successes = batch['success']
                
                # Compute similarities in batch
                with torch.inference_mode():
                    distances = model(rendered_batch, ground_truth_batch)
                    similarities = 1 - distances
                
                # Convert to list and create results
                similarities_list = similarities.cpu().numpy().tolist()
                
                for i, (idx, similarity, success) in enumerate(zip(indices, similarities_list, successes)):
                    if success:
                        results.append({
                            "index": int(idx),
                            "similarity": float(max(0, min(1, similarity))),
                            "success": True
                        })
                    else:
                        # Handle failed preprocessing
                        error_msg = batch.get('error', ['Unknown error'])[i] if 'error' in batch else 'Unknown error'
                        results.append({
                            "index": int(idx),
                            "similarity": 0.0,
                            "success": False,
                            "error": str(error_msg)
                        })
                        
            except Exception as e:
                print(f"Process {process_id}: Error processing batch {batch_idx}: {e}", file=sys.stderr)
                # Add failed results for this batch
                for idx in batch['index']:
                    results.append({
                        "index": int(idx),
                        "similarity": 0.0,
                        "success": False,
                        "error": str(e)
                    })
        
        # Statistics for current process
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        print(f"GPU {gpu_id} Process {process_id} completed: {successful} success, {failed} failed", file=sys.stderr)
        
        # Explicit cleanup before returning
        del model  # Delete the model explicitly
        torch.cuda.empty_cache() if torch.cuda.is_available() else None  # Clear GPU cache
        gc.collect()  # Force garbage collection
        print(f"Process {process_id}: Local resources cleaned up", file=sys.stderr)
        
        return results
        
    except Exception as e:
        print(f"Error: GPU {gpu_id} Process {process_id} initialization failed: {e}", file=sys.stderr)
        return [{
            "index": i,
            "similarity": 0.0,
            "success": False,
            "error": str(e)
        } for i in range(len(image_pairs_chunk))]


def split_data_for_gpus(image_pairs, num_gpus):
    """
    Split image pair data into multiple chunks for different GPUs
    """
    if num_gpus <= 0:
        return [image_pairs]
    
    chunk_size = len(image_pairs) // num_gpus
    remainder = len(image_pairs) % num_gpus
    
    chunks = []
    start_idx = 0
    
    for i in range(num_gpus):
        # First few processes get one extra data point to handle remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunks.append(image_pairs[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='DreamSim worker for batch image similarity computation')
    parser.add_argument('input_file', help='Path to input JSON file containing image pairs')
    parser.add_argument('output_file', help='Path to output JSON file for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 32)')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs to use (default: 8)')
    args = parser.parse_args()
    
    # Multi-GPU mode setup
    num_processes = args.num_gpus
    print(f"Using multi-GPU mode: {num_processes} processes", file=sys.stderr)
    
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)
    
    image_pairs = input_data.get("image_pairs", [])
    if not image_pairs:
        output_error = {"error": "No image pairs provided", "success": False}
        with open(args.output_file, 'w') as f:
            json.dump(output_error, f)
        sys.exit(1)
    print(f"Found {len(image_pairs)} image pairs to process", file=sys.stderr)
    print(f"Using batch size: {args.batch_size}", file=sys.stderr)
    
    try:
        # Multi-GPU parallel processing
        print(f"Starting multi-GPU parallel processing with {num_processes} processes...", file=sys.stderr)
        
        # Split data
        data_chunks = split_data_for_gpus(image_pairs, num_processes)
        print(f"Data splitting completed, chunk sizes: {[len(chunk) for chunk in data_chunks]}", file=sys.stderr)
        
        # Create process pool with improved shutdown handling
        pool = mp.Pool(processes=num_processes)
        try:
            # Create task arguments
            tasks = []
            for i, chunk in enumerate(data_chunks):
                if len(chunk) > 0:  # Only process non-empty chunks
                    gpu_id = i % num_processes  # Simple GPU allocation strategy
                    task_args = (gpu_id, chunk, i+1, num_processes, args.batch_size)
                    tasks.append(task_args)
            
            # Execute tasks in parallel
            print(f"Starting {len(tasks)} GPU processes...", file=sys.stderr)
            chunk_results = pool.starmap(process_gpu_worker, tasks)
            
            print("All GPU processes completed, closing process pool...", file=sys.stderr)
            
        finally:
            # Graceful shutdown with timeout
            pool.close()  # No more work
            print("Waiting for worker processes to finish...", file=sys.stderr)
            pool.join()   # Wait for workers to finish
            print("All worker processes finished", file=sys.stderr)
        
        # Merge all results
        print("Merging all processing results...", file=sys.stderr)

        # Merge all results (indices are already correct from workers)
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        print(f"Results merged successfully, processed {len(results)} image pairs", file=sys.stderr)
        
        # Write results to output file
        print("Computation completed, writing results to file...", file=sys.stderr)
        output = {
            "results": results,
            "success": True
        }
        with open(args.output_file, 'w') as f:
            json.dump(output, f)
        print(f"Results written to {args.output_file}", file=sys.stderr)
        
    except Exception as e:
        output_error = {
            "error": f"Failed to compute DreamSim similarities: {e}",
            "success": False
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_error, f)
        sys.exit(1)


if __name__ == "__main__":
    main() 