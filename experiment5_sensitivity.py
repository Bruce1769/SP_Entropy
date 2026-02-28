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

def evaluate_threshold(input_ids, max_tokens, tokenizer, gt_answer, small_model, large_model, gamma, threshold):
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
        eos_token_id=tokenizer.eos_token_id
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
    
    # We explore a range of thresholds. 
    # threshold = 0.0 means essentially standard SP (no entropy relaxation)
    # very high threshold means almost always relaxed
    thresholds = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    results_by_thresh = {t: {"correct": 0, "tokens": 0, "time": 0.0, "accept_ratio_sum": 0.0, "samples": 0} for t in thresholds}
    
    print(f"\nRunning Experiment 5 on {len(target_lines)} samples across {len(thresholds)} thresholds...")
    
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue
            
        for t in thresholds:
            res = evaluate_threshold(input_ids, args.max_tokens, tokenizer, gt_answer, small_model, large_model, args.gamma, t)
            
            results_by_thresh[t]["correct"] += int(res["correct"])
            results_by_thresh[t]["tokens"] += res["tokens"]
            results_by_thresh[t]["time"] += res["time"]
            results_by_thresh[t]["accept_ratio_sum"] += res["accept_ratio"]
            results_by_thresh[t]["samples"] += 1

    # Aggregate metrics
    acc_list = []
    tps_list = []
    ar_list = []
    
    print("\n" + "="*80)
    print(f"{'Threshold':<10} | {'Accuracy':<10} | {'Tokens/sec':<12} | {'Accept Ratio':<15}")
    print("-" * 80)
    
    for t in thresholds:
        metrics = results_by_thresh[t]
        samples = metrics["samples"]
        if samples == 0:
            continue
            
        acc = metrics["correct"] / samples * 100
        tps = metrics["tokens"] / metrics["time"] if metrics["time"] > 0 else 0
        ar = metrics["accept_ratio_sum"] / samples * 100
        
        acc_list.append(acc)
        tps_list.append(tps)
        ar_list.append(ar)
        
        print(f"{t:<10.1f} | {acc:>5.1f}%     | {tps:>10.1f}   | {ar:>10.1f}%")
    print("=" * 80)
    
    # Plotting: Three metrics on the same X-axis (threshold)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.set_xlabel('Entropy Threshold (τ)', fontsize=12)
    ax1.set_ylabel('Speed (Tokens / sec)', color=color1, fontsize=12)
    ln1 = ax1.plot(thresholds, tps_list, marker='o', color=color1, label='Speed', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Task Accuracy (%)', color=color2, fontsize=12)  
    ln2 = ax2.plot(thresholds, acc_list, marker='s', color=color2, label='Accuracy', linewidth=2, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)

    ax3 = ax1.twinx()
    color3 = 'tab:red'
    # Move ax3 to the right by some offset to not overlap with ax2
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Accept Ratio (%)', color=color3, fontsize=12)
    ln3 = ax3.plot(thresholds, ar_list, marker='^', color=color3, label='Accept Ratio', linewidth=2, linestyle=':')
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Combine legends
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=11)
    
    plt.title('Sensitivity Study: Impact of Entropy Threshold (τ)', fontsize=14)
    fig.tight_layout()
    plt.savefig('experiment5_sensitivity.png', bbox_inches='tight')
    print("Saved plot to experiment5_sensitivity.png")

if __name__ == "__main__":
    main()
