import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import time

from sampling.sampling_utils import format_math_prompt
from sampling.speculative_sampling_entropy_based import speculative_sampling_entropy_based
from experiment1_motivation import extract_answer

MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def evaluate_threshold_strategy(input_ids, max_tokens, tokenizer, gt_answer, small_model, large_model, gamma, threshold, strategy):
    start_time = time.perf_counter()
    
    # Use deterministic seed for comparability
    torch.manual_seed(123)
    
    results = speculative_sampling_entropy_based(
        input_ids, 
        small_model, 
        large_model, 
        max_len=max_tokens, 
        gamma=gamma, 
        entropy_threshold=threshold, 
        eos_token_id=tokenizer.eos_token_id,
        relaxation_strategy=strategy
    )
    
    output_ids, accepted_count, num_steps, _ = results[:4]
    elapsed = time.perf_counter() - start_time
    
    new_tokens = output_ids[0][input_ids.shape[1]:]
    num_generated = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    ans = extract_answer(text)
    correct = (ans == gt_answer) if gt_answer else False
    
    accept_ratio = accepted_count / (num_steps * gamma) if num_steps > 0 else 0.0
    
    return {
        "text": text,
        "correct": correct,
        "tokens": num_generated,
        "time": elapsed,
        "steps": num_steps,
        "accept_ratio": accept_ratio
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--max_tokens', type=int, default=1024)
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
    
    thresholds = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
    strategies = ['sqrt', 'add_0.5']
    
    # Initialize results structure
    results = {
        s: {t: {"correct": 0, "tokens": 0, "time": 0.0, "accept_ratio_sum": 0.0, "samples": 0} for t in thresholds}
        for s in strategies
    }
    
    print(f"\nRunning Ablation on {len(target_lines)} samples, {len(thresholds)} thresholds, {len(strategies)} strategies...")
    
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue
            
        for t in thresholds:
            for s in strategies:
                res = evaluate_threshold_strategy(input_ids, args.max_tokens, tokenizer, gt_answer, small_model, large_model, args.gamma, t, s)
                
                results[s][t]["correct"] += int(res["correct"])
                results[s][t]["tokens"] += res["tokens"]
                results[s][t]["time"] += res["time"]
                results[s][t]["accept_ratio_sum"] += res["accept_ratio"]
                results[s][t]["samples"] += 1

    # Print and Plot
    plot_data = {s: {"acc": [], "ar": []} for s in strategies}
    
    for s in strategies:
        print(f"\n=== Strategy: {s} ===")
        print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Accept Ratio':<15}")
        print("-" * 40)
        for t in thresholds:
            metrics = results[s][t]
            samples = metrics["samples"]
            if samples == 0:
                continue
            
            acc = metrics["correct"] / samples * 100
            ar = metrics["accept_ratio_sum"] / samples * 100
            
            plot_data[s]["acc"].append(acc)
            plot_data[s]["ar"].append(ar)
            
            print(f"{t:<10.1f} | {acc:>5.1f}%     | {ar:>10.1f}%")
            
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accept Ratio Plot
    ax1.plot(thresholds, plot_data['sqrt']["ar"], marker='o', label='sqrt(p/q) [Conservative]', color='blue', linewidth=2)
    ax1.plot(thresholds, plot_data['add_0.5']["ar"], marker='s', label='p/q + 0.5 [Aggressive]', color='red', linewidth=2, linestyle='--')
    ax1.set_title('Acceptance Rate vs. Entropy Threshold')
    ax1.set_xlabel('Entropy Threshold (τ)')
    ax1.set_ylabel('Acceptance Rate (%)')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend()
    
    # Accuracy Plot
    ax2.plot(thresholds, plot_data['sqrt']["acc"], marker='o', label='sqrt(p/q) [Conservative]', color='blue', linewidth=2)
    ax2.plot(thresholds, plot_data['add_0.5']["acc"], marker='s', label='p/q + 0.5 [Aggressive]', color='red', linewidth=2, linestyle='--')
    ax2.set_title('Task Accuracy vs. Entropy Threshold')
    ax2.set_xlabel('Entropy Threshold (τ)')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.legend()
    
    plt.suptitle('Ablation Study: Relaxation Strategies in Speculative Sampling', fontsize=16)
    plt.tight_layout()
    plt.savefig('experiment5_strategy_ablation.png', bbox_inches='tight')
    print("\nSaved plot to experiment5_strategy_ablation.png")

if __name__ == "__main__":
    main()
