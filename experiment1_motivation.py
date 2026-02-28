import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path

from sampling.sampling_utils import format_math_prompt, norm_logits, sample

# Copy of MODELZOO for convenience
MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def extract_answer(text):
    """Extracts content inside \boxed{}"""
    match = re.findall(r'\\boxed{(.*?)}', text)
    if match:
        return match[-1] # Return the last boxed answer
    return None

@torch.no_grad()
def autoregressive_sampling_with_entropy(x: torch.Tensor, model: torch.nn.Module, N: int, 
                                         temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
                                         eos_token_id: int = None):
    n = len(x[0])
    T = n + N
    past_key_values = None
    entropies = []
    generated_tokens = []

    while n < T:
        if past_key_values:
            last_ids = x[:, -1:]
            outputs = model(last_ids, past_key_values=past_key_values, use_cache=True)
        else:
            outputs = model(x)
            
        logits = outputs.logits[:, -1, :]
        probs = norm_logits(logits, temperature, top_k, top_p)
        
        # Calculate entropy (using nats)
        ent = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        entropies.append(ent.item())
        
        past_key_values = outputs.past_key_values
        idx_next = sample(probs, temperature, top_k, top_p)
        
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break
            
        x = torch.cat((x, idx_next), dim=1)
        generated_tokens.append(idx_next.item())
        n += 1
        
    return x, entropies, generated_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50)
    args = parser.parse_args()

    small_model_path = MODELZOO["Qwen2.5-0.5B"]
    large_model_path = MODELZOO["Qwen2.5-7B"]
    
    print("Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(small_model_path, local_files_only=True, trust_remote_code=True)
    
    small_model = AutoModelForCausalLM.from_pretrained(small_model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    large_model = AutoModelForCausalLM.from_pretrained(large_model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    
    small_model.resize_token_embeddings(len(tokenizer))
    large_model.resize_token_embeddings(len(tokenizer))
    
    # Load dataset
    with open('/remote-home/pxl/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/ff5b20257d8185524591543f8ff5993951537bb8/test.jsonl', 'r') as f:
        all_lines = f.readlines()
        
    target_lines = all_lines[:args.num_samples]
    
    results = []
    
    print(f"Running experiment on {len(target_lines)} samples...")
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        gt_answer = extract_answer(data["solution"])
        
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue
            
        # 1. Generate with Small Model
        torch.manual_seed(123)
        small_out, small_entropies, small_gen_tokens = autoregressive_sampling_with_entropy(
            input_ids, small_model, N=2048, temperature=1.0, top_k=20, top_p=0.9, eos_token_id=tokenizer.eos_token_id
        )
        small_text = tokenizer.decode(small_gen_tokens, skip_special_tokens=True)
        small_ans = extract_answer(small_text)
        small_correct = (small_ans == gt_answer) if gt_answer else False
        
        # 2. Generate with Large Model
        torch.manual_seed(123)
        large_out, _, large_gen_tokens = autoregressive_sampling_with_entropy(
            input_ids, large_model, N=2048, temperature=1.0, top_k=20, top_p=0.9, eos_token_id=tokenizer.eos_token_id
        )
        large_text = tokenizer.decode(large_gen_tokens, skip_special_tokens=True)
        large_ans = extract_answer(large_text)
        large_correct = (large_ans == gt_answer) if gt_answer else False
        
        # Find divergence point
        divergence_idx = -1
        for i in range(min(len(small_gen_tokens), len(large_gen_tokens))):
            if small_gen_tokens[i] != large_gen_tokens[i]:
                divergence_idx = i
                break
                
        results.append({
            "idx": idx,
            "small_correct": small_correct,
            "large_correct": large_correct,
            "divergence_idx": divergence_idx,
            "small_entropies": small_entropies,
            "small_gen_len": len(small_gen_tokens)
        })
        
    # Analyze and Plot
    correct_entropies = []
    incorrect_entropies = []
    
    # 统计小模型答错但大模型答对的样本中，偏离点附近的熵
    divergence_entropies = []
    baseline_entropies = []

    for r in results:
        ents = r["small_entropies"]
        if r["small_correct"]:
            correct_entropies.extend(ents)
        else:
            incorrect_entropies.extend(ents)
            
        # 只有在 small_wrong 且 large_correct 时，偏离点才真正代表"能力不足"
        if not r["small_correct"] and r["large_correct"] and r["divergence_idx"] != -1:
            div_idx = r["divergence_idx"]
            # 收集偏离点前后的熵
            window = 5
            start = max(0, div_idx - window)
            end = min(len(ents), div_idx + window + 1)
            
            # 记录相对偏离点的位置的熵 (-5 to +5)
            rel_ents = {}
            for i in range(start, end):
                rel_ents[i - div_idx] = ents[i]
            divergence_entropies.append(rel_ents)
            
    print(f"\n--- Statistics ---")
    print(f"Small model correct: {sum(1 for r in results if r['small_correct'])} / {len(results)}")
    print(f"Large model correct: {sum(1 for r in results if r['large_correct'])} / {len(results)}")
    print(f"Small WRONG & Large CORRECT: {sum(1 for r in results if not r['small_correct'] and r['large_correct'])}")
    
    print(f"\nAverage Entropy (Small Correct): {np.mean(correct_entropies):.4f}")
    print(f"Average Entropy (Small Incorrect): {np.mean(incorrect_entropies):.4f}")
    
    # Plot 1: Overall Entropy Distribution
    plt.figure(figsize=(10, 5))
    plt.hist(correct_entropies, bins=50, alpha=0.5, label='Small Correct', density=True)
    plt.hist(incorrect_entropies, bins=50, alpha=0.5, label='Small Incorrect', density=True)
    plt.title('Token Entropy Distribution: Correct vs Incorrect Generations')
    plt.xlabel('Entropy')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('entropy_dist.png')
    print("Saved entropy_dist.png")

    # Plot 2: Entropy around Divergence Point
    if divergence_entropies:
        agg_div = {}
        for re_ents in divergence_entropies:
            for offset, ent in re_ents.items():
                if offset not in agg_div:
                    agg_div[offset] = []
                agg_div[offset].append(ent)
                
        offsets = sorted(list(agg_div.keys()))
        mean_ents = [np.mean(agg_div[o]) for o in offsets]
        
        plt.figure(figsize=(10, 5))
        plt.plot(offsets, mean_ents, marker='o', color='red', linewidth=2)
        plt.axvline(x=0, color='black', linestyle='--', label='Divergence Point')
        plt.title('Average Entropy Around Divergence Point (Small Wrong, Large Correct)')
        plt.xlabel('Relative Token Position from Divergence')
        plt.ylabel('Average Entropy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('divergence_entropy.png')
        print("Saved divergence_entropy.png")
    else:
        print("Not enough divergence samples to plot.")

if __name__ == "__main__":
    main()
