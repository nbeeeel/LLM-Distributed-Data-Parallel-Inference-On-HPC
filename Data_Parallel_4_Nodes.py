import os
import torch
import torch.distributed as dist
import time
import socket
from transformers import AutoTokenizer, AutoModelForCausalLM

# Config
MODEL_PATH = "/mnt/lustre/user/llama3.2-3b-instruct"
MAX_TOKENS = 100
TEMPERATURE = 0.7

def log(message):
    hostname = socket.gethostname()
    rank = int(os.environ.get('RANK', 0))
    print(f"[{hostname}:R{rank}] {message}", flush=True)

def get_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"{allocated:.1f}GB/{total:.1f}GB"
    return "No CUDA"

def setup_distributed():
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')

    # Set GPU device
    local_rank = int(os.environ.get('SLURM_LOCALID', 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')

    log(f"Setting up distributed: rank={rank}, world_size={world_size}")
    log(f"Using GPU {local_rank}, device={device}")

    try:
        # Initialize with NCCL backend for GPU communication
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size
        )
        log("NCCL distributed setup complete")
        return rank, world_size, device
    except Exception as e:
        log(f"Distributed setup failed: {e}")
        return None, None, None

def load_model(device):
    """Load full model on each node"""
    log(f"Memory before loading: {get_memory_usage()}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load full model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )

    log(f"Memory after loading: {get_memory_usage()}")
    return model, tokenizer

def generate_responses(model, tokenizer, prompts, device):
    """Generate responses for assigned prompts"""
    results = []

    for i, prompt in enumerate(prompts):
        log(f"Processing prompt {i+1}/{len(prompts)}: '{prompt[:50]}...'")

        start_time = time.time()

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = full_text[len(prompt):].strip()

        generation_time = time.time() - start_time
        output_tokens = len(tokenizer.encode(response))

        result = {
            'prompt': prompt,
            'response': response,
            'tokens': output_tokens,
            'time': generation_time,
            'speed': output_tokens / generation_time if generation_time > 0 else 0
        }

        results.append(result)
        log(f"Completed prompt {i+1}: {output_tokens} tokens, {result['speed']:.2f} tok/s")

    return results

def main():
    # Setup distributed environment
    rank, world_size, device = setup_distributed()

    if rank is None:
        log("Failed to setup distributed environment")
        return

    # Load model on each node
    log("Loading full model...")
    model, tokenizer = load_model(device)

    # Test prompts
    all_prompts = [
        "What is artificial intelligence?",
        "How do neural networks work?",
        "Explain machine learning simply.",
        "What is distributed computing?",
        "How do transformers work?"
    ]

    # Split prompts among nodes
    prompts_per_rank = len(all_prompts) // world_size
    start_idx = rank * prompts_per_rank

    if rank == world_size - 1:  # Last rank gets remaining prompts
        end_idx = len(all_prompts)
    else:
        end_idx = start_idx + prompts_per_rank

    my_prompts = all_prompts[start_idx:end_idx]

    log(f"Assigned prompts {start_idx}-{end_idx-1} ({len(my_prompts)} prompts)")

    # Sync all nodes before starting
    dist.barrier()

    if rank == 0:
        print("\n" + "="*60)
        print("DATA PARALLEL INFERENCE TEST")
        print("="*60)
        print(f"Model: LLaMA 3.2-3B (Full replication)")
        print(f"Nodes: {world_size}")
        print(f"Total prompts: {len(all_prompts)}")
        print(f"Prompts per node: ~{prompts_per_rank}")
        print("="*60)

    # Process assigned prompts
    start_time = time.time()
    results = generate_responses(model, tokenizer, my_prompts, device)
    processing_time = time.time() - start_time

    # Wait for all nodes to complete
    dist.barrier()

    # Gather results from all nodes
    all_results = [None] * world_size
    dist.all_gather_object(all_results, results)

    # Print results
    if rank == 0:
        print("\n" + "="*80)
        print("DATA PARALLEL INFERENCE RESULTS")
        print("="*80)

        total_tokens = 0
        total_time = 0

        for node_rank, node_results in enumerate(all_results):
            if node_results:
                node_tokens = sum(r['tokens'] for r in node_results)
                node_time = sum(r['time'] for r in node_results)

                print(f"\n{'='*20} NODE {node_rank} RESULTS {'='*20}")
                print(f"Processed {len(node_results)} prompts:")

                for i, result in enumerate(node_results):
                    print(f"\n--- Prompt {i+1} ---")
                    print(f"Q: {result['prompt']}")
                    print(f"A: {result['response']}")
                    print(f"Stats: {result['tokens']} tokens, {result['speed']:.2f} tok/s, {result['time']:.2f}s")

                print(f"\nNode {node_rank} Summary: {node_tokens} tokens, {node_time:.2f}s")
                total_tokens += node_tokens
                total_time += node_time

        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        print(f"Total prompts processed: {len(all_prompts)}")
        print(f"Total tokens generated: {total_tokens}")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Parallel processing time: {processing_time:.2f}s")
        print(f"Average generation speed: {total_tokens/total_time:.2f} tok/s")
        print(f"Speedup achieved: {total_time/processing_time:.2f}x")
        print("="*80)
        print("DATA PARALLEL TEST COMPLETED SUCCESSFULLY!")
        print("="*80)

    else:
        # Worker nodes just log their completion
        my_tokens = sum(r['tokens'] for r in results)
        log(f"Completed {len(results)} prompts, {my_tokens} tokens in {processing_time:.2f}s")

    # Cleanup
    log(f"Final memory usage: {get_memory_usage()}")
    dist.destroy_process_group()
    log("Data parallel test completed")

if __name__ == "__main__":
    main()