import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np

from sampling.sampling_utils import format_math_prompt
from sampling.speculative_sampling_entropy_based import speculative_sampling_entropy_based

MODELZOO = {
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--gamma', type=int, default=4)
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
    
    agg_stats = {
        "total_evaluated": 0,
        "low_entropy_evaluated": 0,
        "high_entropy_evaluated": 0,
        "low_entropy_accepted": 0,
        "high_entropy_accepted": 0,
        "rescued_by_relaxation": 0
    }
    
    print(f"\nRunning Experiment 4 on {len(target_lines)} samples...")
    for idx, line in enumerate(tqdm(target_lines)):
        data = json.loads(line)
        prompt = format_math_prompt(data["problem"])
        input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(small_model.device)
        
        if input_ids.shape[1] > 2048:
            continue

        torch.manual_seed(123 + idx)
        results = speculative_sampling_entropy_based(
            input_ids, 
            small_model, 
            large_model, 
            max_len=args.max_tokens, 
            gamma=args.gamma, 
            entropy_threshold=args.entropy_threshold, 
            eos_token_id=tokenizer.eos_token_id,
            return_stats=True
        )
        
        # Results tuple: prefix, accepted_count, num_steps, all_entropies, stats
        if len(results) >= 5:
            stats = results[4]
            for k in agg_stats:
                agg_stats[k] += stats.get(k, 0)

    print("\n" + "="*60)
    print("EXPERIMENT 4: VERIFICATION MECHANISM ANALYSIS")
    print("="*60)
    
    tot_eval = agg_stats["total_evaluated"]
    if tot_eval == 0:
        print("No tokens evaluated.")
        return
        
    low_eval = agg_stats["low_entropy_evaluated"]
    high_eval = agg_stats["high_entropy_evaluated"]
    low_acc = agg_stats["low_entropy_accepted"]
    high_acc = agg_stats["high_entropy_accepted"]
    rescued = agg_stats["rescued_by_relaxation"]
    
    tot_acc = low_acc + high_acc
    
    print(f"Total Draft Tokens Evaluated: {tot_eval}")
    print("-" * 60)
    print(f"Low Entropy Tokens (<= {args.entropy_threshold}):")
    print(f"  - Evaluated: {low_eval} ({low_eval/tot_eval*100:.1f}% of total)")
    print(f"  - Accepted:  {low_acc} (Acceptance Rate: {low_acc/low_eval*100:.1f}%)")
    print(f"  - Rescued by Relaxation: {rescued} ({rescued/low_acc*100:.1f}% of low-ent accepted)")
    print("-" * 60)
    print(f"High Entropy Tokens (> {args.entropy_threshold}):")
    print(f"  - Evaluated: {high_eval} ({high_eval/tot_eval*100:.1f}% of total)")
    print(f"  - Accepted:  {high_acc} (Acceptance Rate: {high_acc/high_eval*100:.1f}%)")
    print("=" * 60)
    
    overall_acc_rate = tot_acc / tot_eval
    estimated_std_acc_rate = (tot_acc - rescued) / tot_eval
    
    print(f"Overall Effective Acceptance Rate: {overall_acc_rate*100:.1f}%")
    print(f"Estimated Standard SP Accept Rate: {estimated_std_acc_rate*100:.1f}%")
    print(f"Absolute Accept Rate Boost:        +{(overall_acc_rate - estimated_std_acc_rate)*100:.1f}%")
    print("=" * 60)
    
    # Visualization 1: Acceptance Rate Comparison (Low vs High Entropy)
    plt.figure(figsize=(8, 6))
    categories = ['Low Entropy\n(Relaxed path)', 'High Entropy\n(Strict path)']
    eval_counts = [low_eval, high_eval]
    acc_counts = [low_acc, high_acc]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, eval_counts, width, label='Evaluated', color='lightgray')
    rects2 = ax.bar(x + width/2, acc_counts, width, label='Accepted', color='#1f77b4')
    
    # Highlight the rescued portion
    ax.bar(x[0] + width/2, rescued, width, bottom=(low_acc - rescued), label='Rescued by Relaxation', color='#ff7f0e', hatch='//')

    ax.set_ylabel('Number of Tokens')
    ax.set_title('Draft Token Verification: Low vs High Entropy Regions')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    def autolabel(rects, is_acc=False, idx=0):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            text = f'{height}'
            if is_acc:
                rate = height / eval_counts[i] * 100
                text += f'\n({rate:.1f}%)'
            ax.annotate(text,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    autolabel(rects1)
    autolabel(rects2, is_acc=True)
    
    fig.tight_layout()
    plt.savefig('experiment4_acceptance_breakdown.png', bbox_inches='tight')
    print("Saved plot to experiment4_acceptance_breakdown.png")
    
    # Visualization 2: Pie chart of Accepted Tokens Contribution
    plt.figure(figsize=(7, 7))
    labels = ['Low Entropy (Strictly Passed)', 'Low Entropy (Rescued)', 'High Entropy (Passed)']
    sizes = [low_acc - rescued, rescued, high_acc]
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    explode = (0, 0.1, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.title('Composition of Total Accepted Tokens')
    plt.savefig('experiment4_accepted_composition.png', bbox_inches='tight')
    print("Saved plot to experiment4_accepted_composition.png")

if __name__ == "__main__":
    main()
