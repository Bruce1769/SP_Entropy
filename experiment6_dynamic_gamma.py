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
from sampling.speculative_sampling_entropy_based import speculative_sampling_entropy_based
from experiment1_motivation import extract_answer

MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def evaluate_method(method_name, input_ids, max_tokens, tokenizer, gt_answer, **kwargs):
    start_time = time.perf_counter()
    torch.manual_seed(123)
    
    if method_name == 'AS_large':
        output_ids = autoregressive_sampling(input_ids, **kwargs)
        num_steps = output_ids.shape[1] - input_ids.shape[1]
        accepted_count = 0
        total_drafted = 0
    else:
        results = speculative_sampling_entropy_based(input_ids, max_len=max_tokens, **kwargs)
        output_ids = results[0]
        accepted_count = results[1]
        num_steps = results[2]
        # Calculate actual total drafted tokens. In our modified script, 
        # total_drafted isn't directly returned but we can approximate it or modify the return.
        # Wait, I didn't return total_drafted from the function!
        # Let's just estimate it from the gamma parameter if fixed, 
        # or we can modify the function to return it. 
        # Actually, let's just use a trick: 
        # The accepted count / accept_ratio = total_drafted if we printed it.
        # But we need it precisely. Let's assume we can get it from the stats dictionary if return_stats=True.
        # Yes! We added return_stats. So stats['total_evaluated'] is exactly total_drafted!
        stats = results[4] if len(results) >= 5 else {}
        total_drafted = stats.get("total_evaluated", num_steps * kwargs.get('gamma', 4))
        
    elapsed = time.perf_counter() - start_time
    
    new_tokens = output_ids[0][input_ids.shape[1]:]
    num_generated = len(new_tokens)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    ans = extract_answer(text)
    correct = (ans == gt_answer) if gt_answer else False
    
    accept_ratio = accepted_count / total_drafted if total_drafted > 0 else 0.0
    avg_gamma = total_drafted / num_steps if num_steps > 0 else 0.0
    
    return {
        "text": text,
        "correct": correct,
        "tokens": num_generated,
        "time": elapsed,
        "steps": num_steps,
        "accept_ratio": accept_ratio,
        "avg_gamma": avg_gamma,
        "total_drafted": total_drafted,
        "large_calls": num_steps if method_name != 'AS_large' else num_generated
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--fixed_gamma', type=int, default=4)
    parser.add_argument('--dynamic_max_gamma', type=int, default=12)
    parser.add_argument('--entropy_threshold', type=float, default=2.5)
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
    
    methods = {
        "AS_large": {"correct": 0, "tokens": 0, "time": 0.0, "steps": 0, "total_drafted": 0, "large_calls": 0},
        "SP_fixed": {"correct": 0, "tokens": 0, "time": 0.0, "steps": 0, "total_drafted": 0, "large_calls": 0, "accepted": 0},
        "SP_dynamic": {"correct": 0, "tokens": 0, "time": 0.0, "steps": 0, "total_drafted": 0, "large_calls": 0, "accepted": 0},
    }
    
    print(f"\nRunning Experiment 6 on {len(target_lines)} samples...")
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue

        # AS_large
        res = evaluate_method('AS_large', input_ids, args.max_tokens, tokenizer, gt_answer, model=large_model, N=args.max_tokens, eos_token_id=tokenizer.eos_token_id)
        for k in ["correct", "tokens", "time", "steps", "total_drafted", "large_calls"]:
            methods["AS_large"][k] += int(res[k]) if k == "correct" else res[k]

        # SP_fixed (Entropy-based, fixed gamma)
        res = evaluate_method('SP_fixed', input_ids, args.max_tokens, tokenizer, gt_answer, 
                              approx_model_raw=small_model, target_model_raw=large_model, 
                              gamma=args.fixed_gamma, entropy_threshold=args.entropy_threshold, 
                              eos_token_id=tokenizer.eos_token_id, return_stats=True)
        for k in ["correct", "tokens", "time", "steps", "total_drafted", "large_calls"]:
            methods["SP_fixed"][k] += int(res[k]) if k == "correct" else res[k]
        methods["SP_fixed"]["accepted"] += res["accept_ratio"] * res["total_drafted"]

        # SP_dynamic (Entropy-based, dynamic gamma)
        res = evaluate_method('SP_dynamic', input_ids, args.max_tokens, tokenizer, gt_answer, 
                              approx_model_raw=small_model, target_model_raw=large_model, 
                              gamma=args.dynamic_max_gamma, entropy_threshold=args.entropy_threshold, 
                              eos_token_id=tokenizer.eos_token_id, dynamic_gamma=True, return_stats=True)
        for k in ["correct", "tokens", "time", "steps", "total_drafted", "large_calls"]:
            methods["SP_dynamic"][k] += int(res[k]) if k == "correct" else res[k]
        methods["SP_dynamic"]["accepted"] += res["accept_ratio"] * res["total_drafted"]

    # Calculate final metrics
    names = ["AS_large", f"SP_entropy (Fixed γ={args.fixed_gamma})", f"SP_dynamic (Max γ={args.dynamic_max_gamma})"]
    keys = ["AS_large", "SP_fixed", "SP_dynamic"]
    
    accuracies = []
    speeds = []
    avg_gammas = []
    accept_ratios = []

    print("\n" + "="*95)
    print(f"{'Method':<30} | {'Accuracy':<8} | {'Tokens/sec':<10} | {'Avg γ':<6} | {'Accept Ratio':<12}")
    print("-" * 95)
    
    for k, name in zip(keys, names):
        m = methods[k]
        acc = m["correct"] / len(target_lines) * 100
        tps = m["tokens"] / m["time"] if m["time"] > 0 else 0
        avg_g = m["total_drafted"] / m["steps"] if m["steps"] > 0 else 0
        ar = (m["accepted"] / m["total_drafted"] * 100) if m["total_drafted"] > 0 else 0
        
        accuracies.append(acc)
        speeds.append(tps)
        avg_gammas.append(avg_g)
        accept_ratios.append(ar)
        
        ar_str = f"{ar:.1f}%" if k != "AS_large" else "N/A"
        gamma_str = f"{avg_g:.2f}" if k != "AS_large" else "N/A"
        
        print(f"{name:<30} | {acc:>6.1f}%   | {tps:>10.1f} | {gamma_str:>6} | {ar_str:>12}")
    print("=" * 95)

    # Plot 1: Speed Comparison
    plt.figure(figsize=(10, 6))
    colors = ['#7f7f7f', '#1f77b4', '#ff7f0e']
    bars = plt.bar(names, speeds, color=colors, width=0.5)
    plt.title('Generation Speed Comparison (Tokens/sec)', fontsize=14)
    plt.ylabel('Tokens / sec (Higher is better)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontweight='bold')
        
    plt.savefig('experiment6_speed_comparison.png', bbox_inches='tight')
    print("Saved plot to experiment6_speed_comparison.png")

    # Plot 2: Average Gamma Comparison
    plt.figure(figsize=(8, 5))
    plot_names = names[1:] # Exclude AS_large
    plot_gammas = avg_gammas[1:]
    bars = plt.bar(plot_names, plot_gammas, color=colors[1:], width=0.4)
    plt.title('Average Draft Length (γ) per Step', fontsize=14)
    plt.ylabel('Average γ')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add horizontal lines for the maximums
    plt.axhline(y=args.fixed_gamma, color='blue', linestyle=':', label=f'Fixed Max ({args.fixed_gamma})')
    plt.axhline(y=args.dynamic_max_gamma, color='orange', linestyle=':', label=f'Dynamic Max ({args.dynamic_max_gamma})')
    plt.legend()
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

    plt.savefig('experiment6_avg_gamma.png', bbox_inches='tight')
    print("Saved plot to experiment6_avg_gamma.png")

if __name__ == "__main__":
    main()
