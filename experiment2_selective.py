import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re

from sampling.sampling_utils import format_math_prompt
from experiment1_motivation import autoregressive_sampling_with_entropy, extract_answer
from sampling.entropy_selective_sampling import entropy_selective_sampling

MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--entropy_threshold', type=float, default=0.1, help="Threshold to switch to large model")
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
    
    results = []
    
    print(f"Running Experiment 2 on {len(target_lines)} samples (Entropy Threshold: {args.entropy_threshold})...")
    
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue
            
        # 1. Pure Small Model
        torch.manual_seed(123)
        small_out, _, small_gen_tokens = autoregressive_sampling_with_entropy(
            input_ids, small_model, N=2048, temperature=1.0, top_k=20, top_p=0.9, eos_token_id=tokenizer.eos_token_id
        )
        small_text = tokenizer.decode(small_gen_tokens, skip_special_tokens=True)
        small_correct = (extract_answer(small_text) == gt_answer) if gt_answer else False
        
        # 2. Pure Large Model
        torch.manual_seed(123)
        large_out, _, large_gen_tokens = autoregressive_sampling_with_entropy(
            input_ids, large_model, N=2048, temperature=1.0, top_k=20, top_p=0.9, eos_token_id=tokenizer.eos_token_id
        )
        large_text = tokenizer.decode(large_gen_tokens, skip_special_tokens=True)
        large_correct = (extract_answer(large_text) == gt_answer) if gt_answer else False
        
        # 3. Entropy-Selective Model
        torch.manual_seed(123)
        sel_out, large_calls, total_steps = entropy_selective_sampling(
            input_ids, small_model, large_model, N=2048, 
            entropy_threshold=args.entropy_threshold,
            temperature=1.0, top_k=20, top_p=0.9, eos_token_id=tokenizer.eos_token_id
        )
        sel_text = tokenizer.decode(sel_out[0][input_ids.shape[1]:], skip_special_tokens=True)
        sel_correct = (extract_answer(sel_text) == gt_answer) if gt_answer else False
        
        results.append({
            "idx": idx,
            "small_correct": small_correct,
            "large_correct": large_correct,
            "sel_correct": sel_correct,
            "large_calls": large_calls,
            "total_steps": total_steps,
            "large_call_ratio": large_calls / total_steps if total_steps > 0 else 0
        })

    # Print Results
    small_acc = sum(1 for r in results if r['small_correct']) / len(results)
    large_acc = sum(1 for r in results if r['large_correct']) / len(results)
    sel_acc = sum(1 for r in results if r['sel_correct']) / len(results)
    avg_large_call_ratio = np.mean([r['large_call_ratio'] for r in results])
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: ENTROPY-SELECTIVE INTERVENTION RESULTS")
    print("="*50)
    print(f"Total Samples: {len(results)}")
    print(f"Entropy Threshold: {args.entropy_threshold}")
    print("-"*50)
    print(f"[AS_small] Pure 0.5B Accuracy:       {small_acc*100:.1f}%")
    print(f"[AS_large] Pure 7.0B Accuracy:       {large_acc*100:.1f}%")
    print(f"[Selective] Entropy-based Accuracy:  {sel_acc*100:.1f}%")
    print("-"*50)
    print(f"Average 7B calls in Selective model: {avg_large_call_ratio*100:.1f}% of total tokens")
    print("="*50)
    
    # Save results to plot
    methods = ['Pure 0.5B', f'Entropy-Selective\n({avg_large_call_ratio*100:.1f}% 7B calls)', 'Pure 7B']
    accuracies = [small_acc*100, sel_acc*100, large_acc*100]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=colors, width=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.title('Accuracy Comparison: Is Large Model only needed at High Entropy?', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, max(accuracies) + 15)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('experiment2_results.png', bbox_inches='tight')
    print("Saved plot to experiment2_results.png")

if __name__ == "__main__":
    main()
