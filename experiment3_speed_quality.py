import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import time

from sampling.sampling_utils import format_math_prompt
from sampling.autoregressive_sampling import autoregressive_sampling
from sampling.speculative_sampling import speculative_sampling
from sampling.speculative_sampling_entropy_based import speculative_sampling_entropy_based
from experiment1_motivation import extract_answer

MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def evaluate_method(method_fn, input_ids, max_tokens, tokenizer, gt_answer, **kwargs):
    start_time = time.perf_counter()
    
    # Handle random seeds for reproducibility between methods
    torch.manual_seed(123)
    
    if method_fn.__name__ == 'autoregressive_sampling':
        output_ids = method_fn(input_ids, **kwargs)
        num_steps = output_ids.shape[1] - input_ids.shape[1]
    else:
        results = method_fn(input_ids, max_len=max_tokens, **kwargs)
        output_ids = results[0]
        num_steps = results[2]
        
    elapsed = time.perf_counter() - start_time
    
    new_tokens = output_ids[0][input_ids.shape[1]:]
    num_generated = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    ans = extract_answer(text)
    correct = (ans == gt_answer) if gt_answer else False
    
    return {
        "text": text,
        "correct": correct,
        "tokens": num_generated,
        "time": elapsed,
        "steps": num_steps
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--entropy_threshold', type=float, default=2.5)
    parser.add_argument('--max_tokens', type=int, default=2048)
    args = parser.parse_args()

    small_model_path = MODELZOO["Qwen2.5-0.5B"]
    large_model_path = MODELZOO["Qwen2.5-7B"]
    
    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(small_model_path, local_files_only=True, trust_remote_code=True)
    
    small_model = AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    large_model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    
    small_model.resize_token_embeddings(len(tokenizer))
    large_model.resize_token_embeddings(len(tokenizer))
    
    with open('/remote-home/pxl/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/ff5b20257d8185524591543f8ff5993951537bb8/test.jsonl', 'r') as f:
        all_lines = f.readlines()
        
    target_lines = all_lines[:args.num_samples]
    
    metrics = {
        "AS_large": {"correct": 0, "tokens": 0, "time": 0.0, "large_calls": 0},
        "AS_small": {"correct": 0, "tokens": 0, "time": 0.0, "large_calls": 0},
        "SP": {"correct": 0, "tokens": 0, "time": 0.0, "large_calls": 0},
        "SP_entropy": {"correct": 0, "tokens": 0, "time": 0.0, "large_calls": 0},
    }
    
    print(f"\nRunning Experiment 3 on {len(target_lines)} samples...")
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue

        # AS_large
        res = evaluate_method(autoregressive_sampling, input_ids, args.max_tokens, tokenizer, gt_answer, model=large_model, N=args.max_tokens, eos_token_id=tokenizer.eos_token_id)
        metrics["AS_large"]["correct"] += int(res["correct"])
        metrics["AS_large"]["tokens"] += res["tokens"]
        metrics["AS_large"]["time"] += res["time"]
        metrics["AS_large"]["large_calls"] += res["tokens"]  # 1 call per token

        # AS_small
        res = evaluate_method(autoregressive_sampling, input_ids, args.max_tokens, tokenizer, gt_answer, model=small_model, N=args.max_tokens, eos_token_id=tokenizer.eos_token_id)
        metrics["AS_small"]["correct"] += int(res["correct"])
        metrics["AS_small"]["tokens"] += res["tokens"]
        metrics["AS_small"]["time"] += res["time"]
        metrics["AS_small"]["large_calls"] += 0

        # SP
        res = evaluate_method(speculative_sampling, input_ids, args.max_tokens, tokenizer, gt_answer, approx_model_raw=small_model, target_model_raw=large_model, gamma=args.gamma, eos_token_id=tokenizer.eos_token_id)
        metrics["SP"]["correct"] += int(res["correct"])
        metrics["SP"]["tokens"] += res["tokens"]
        metrics["SP"]["time"] += res["time"]
        metrics["SP"]["large_calls"] += (1 + res["steps"]) # 1 prefill + 1 per verify step

        # SP_entropy
        res = evaluate_method(speculative_sampling_entropy_based, input_ids, args.max_tokens, tokenizer, gt_answer, approx_model_raw=small_model, target_model_raw=large_model, gamma=args.gamma, entropy_threshold=args.entropy_threshold, eos_token_id=tokenizer.eos_token_id)
        metrics["SP_entropy"]["correct"] += int(res["correct"])
        metrics["SP_entropy"]["tokens"] += res["tokens"]
        metrics["SP_entropy"]["time"] += res["time"]
        metrics["SP_entropy"]["large_calls"] += (1 + res["steps"])

    # Calculate final averages
    methods = ["AS_large", "AS_small", "SP", "SP_entropy"]
    labels = ["AS_large (7B)", "AS_small (0.5B)", f"SP (γ={args.gamma})", f"SP_entropy (γ={args.gamma})"]
    
    accuracies = []
    speeds = []
    call_ratios = []

    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Accuracy':<10} | {'Tokens/sec':<12} | {'7B Calls/Token':<15}")
    print("-" * 80)
    
    for m in methods:
        total_samples = len(target_lines)
        acc = metrics[m]["correct"] / total_samples * 100
        tps = metrics[m]["tokens"] / metrics[m]["time"] if metrics[m]["time"] > 0 else 0
        calls_per_token = metrics[m]["large_calls"] / metrics[m]["tokens"] if metrics[m]["tokens"] > 0 else 0
        
        accuracies.append(acc)
        speeds.append(tps)
        call_ratios.append(calls_per_token)
        
        print(f"{m:<20} | {acc:>5.1f}%     | {tps:>10.1f}   | {calls_per_token:>10.3f}")
    print("=" * 80)

    # Plot 1: Speed vs Accuracy (Scatter Plot)
    plt.figure(figsize=(10, 6))
    colors = ['#d62728', '#1f77b4', '#7f7f7f', '#2ca02c']
    markers = ['s', 'o', 'x', '*']
    
    for i in range(len(methods)):
        plt.scatter(speeds[i], accuracies[i], color=colors[i], marker=markers[i], s=200, label=labels[i])
        
    plt.title('Speed vs Accuracy Comparison', fontsize=14)
    plt.xlabel('Speed (Tokens / sec) ➔', fontsize=12)
    plt.ylabel('Accuracy (%) ➔', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    # Add an ideal zone annotation
    plt.annotate('Ideal Zone\n(Fast & Accurate)', xy=(max(speeds)*0.95, max(accuracies)), 
                 xycoords='data', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="none", alpha=0.3),
                 ha='center', va='center')
                 
    plt.savefig('experiment3_speed_vs_accuracy.png', bbox_inches='tight')
    print("Saved plot to experiment3_speed_vs_accuracy.png")

    # Plot 2: Large Model Calls Per Token (Bar Chart)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, call_ratios, color=colors, width=0.5)
    plt.title('Large Model (7B) Calls per Token Generated', fontsize=14)
    plt.ylabel('7B Calls / Token (Lower is better) ➔', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('experiment3_calls_per_token.png', bbox_inches='tight')
    print("Saved plot to experiment3_calls_per_token.png")

if __name__ == "__main__":
    main()
