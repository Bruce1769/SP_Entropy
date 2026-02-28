
from tokenize import generate_tokens
# from test_entropy_sampling import generated_text
import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_entropy_based
from globals import Decoder
import json
from  tqdm import tqdm
from datasets import load_dataset
from sampling.sampling_utils import format_math_prompt, sample
from transformers import DynamicCache
from utils import visualize_token_entropy
import matplotlib.pyplot as plt
from IPython import embed
from pathlib import Path
# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "Qwen2.5-0.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    "Qwen2.5-1.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "Qwen2.5-3B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "Qwen2.5-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "Qwen2.5-32B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "Qwen2.5-Math-1.5B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots/aafeb0fc6f22cbf0eaeed126eff8be45b0360a35",
    "Qwen2.5-Math-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d",
    "Qwen2.5-Coder-7B": "/remote-home/pxl/.cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-7B/snapshots/0396a76181e127dfc13e5c5ec48a8cee09938b02",
}

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Suggest at least five related search terms to \"Mạng neural nhân tạo\".")
    parser.add_argument('--approx_model_name', type=str, default="Qwen2.5-0.5B")
    parser.add_argument('--target_model_name', type=str, default="Qwen2.5-7B")
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=4096, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=10, help='guess time.')
    parser.add_argument('--entropy_threshold', '-e', type=float, default=2.5, help='entropy threshold (higher = accept more, SP_entropy faster).')
    parser.add_argument('--top_k', type=int, default=20, help='top-k sampling parameter.')
    parser.add_argument('--top_p', type=float, default=0.9, help='top-p sampling parameter.')
    parser.add_argument('--start', type=int, default=0, help='start index for benchmarking.')
    parser.add_argument('--end', type=int, default=1, help='end index for benchmarking.')
    parser.add_argument('--output_dir', type=str, default="generated_texts", help='directory to save generated txt files.')
    parser.add_argument('--wandb_mode', type=str, default="online", choices=["online", "offline", "disabled"], help='wandb mode.')
    args = parser.parse_args()
    return args


def benchmark(fn, info, tokenizer, *args, start_idx=0, end_idx=5, enable_wandb=True, **kwargs):
    import json
    
    total_tokens = 0
    total_accepted = 0
    total_steps = 0
    generated_texts = []
    all_sample_entropies = [] # 存储每个样本的熵列表

    with open('/remote-home/pxl/.cache/huggingface/hub/datasets--HuggingFaceH4--MATH-500/snapshots/ff5b20257d8185524591543f8ff5993951537bb8/test.jsonl', 'r') as file: # 路径保持原样
        all_lines = file.readlines()

    actual_end = min(end_idx, len(all_lines))
    target_lines = all_lines[start_idx:actual_end]

    with contexttimer.Timer() as t:
        with tqdm(total=len(target_lines), desc=f"{info}") as pbar:
            for line in target_lines:
                data = json.loads(line)
                # print(data["problem"])
                apply_math_prompt = format_math_prompt(data["problem"])
                # add_generation_prompt=True 确保生成部分的特殊标记被添加
                input_ids = tokenizer.apply_chat_template(apply_math_prompt,add_generation_prompt=True, return_tensors='pt')
                
                
                if input_ids.shape[-1] > 4096:
                    pbar.update(1)
                    continue

                # 将 input_ids 放到模型所在设备，避免 device_map="auto" 时分片导致设备不一致
                model = args[0] if args else None
                if model is not None and hasattr(model, 'parameters'):
                    device = next(model.parameters()).device
                    input_ids = input_ids.to(device)

                # 推理
                results = fn(input_ids, *args, **kwargs, eos_token_id=tokenizer.eos_token_id)
                
                # 统一解析返回结果
                entropies = None
                if isinstance(results, tuple):
                    output_ids = results[0]
                    total_accepted += results[1]
                    total_steps += results[2]
                    if len(results) == 4: # 基于熵的投机采样
                        entropies = results[3]
                else:
                    output_ids = results
                
                # 保存文本
                new_tokens = output_ids[0][input_ids.shape[-1]:]
                gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                generated_texts.append(gen_text)
                all_sample_entropies.append(entropies) # 如果没有熵，则存入 None

                total_tokens += new_tokens.shape[-1]
                pbar.update(1)
                    
    # Log 指标 (保持原有逻辑)
    tokens_per_sec = total_tokens / t.elapsed if t.elapsed > 0 else 0
    if enable_wandb:
        try:
            wandb.log({f"{info}/tokens_per_sec": tokens_per_sec})
        except Exception as e:
            print(f"[WARN] wandb.log failed for {info}: {e}")
    
    return generated_texts, all_sample_entropies # 返回文本和熵


def save_method_texts(output_dir, method_name, texts, start_idx):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / f"{method_name}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(texts):
            prompt_idx = start_idx + i
            f.write(f"===== Prompt {prompt_idx} | Method: {method_name} =====\n")
            f.write(text.strip() + "\n\n")

    return str(file_path)

def generate(approx_model_name, target_model_name, num_tokens=4096, gamma = 4, entropy_threshold=1.0, start_idx=0, end_idx=5,
             random_seed = None, output_dir="generated_texts", enable_wandb=True, top_k=20, top_p=0.9):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    approx_path = MODELZOO.get(approx_model_name)
    
    target_path = MODELZOO.get(target_model_name)
    
    tokenizer = AutoTokenizer.from_pretrained(approx_path, local_files_only=True, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_path} \n {target_path}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_path,     
                                                       torch_dtype=torch.bfloat16,
                                                       device_map="auto",
                                                        local_files_only=True, trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_path, torch_dtype=torch.bfloat16,
                                                       device_map="auto",
                                                        local_files_only=True, trust_remote_code=True)
    print("finish loading models")

    # --- ADD THIS FIX ---
    # Resize token embeddings to match the tokenizer's vocab size exactly
    small_model.resize_token_embeddings(len(tokenizer))
    large_model.resize_token_embeddings(len(tokenizer))
    # --------------------
    
    # 准备 WandB 表格（可选）
    # 我们将记录：Prompt索引, 方法名称, Gamma, 阈值, 生成的文本
    results_table = None
    if enable_wandb:
        results_table = wandb.Table(columns=["Prompt_Index", "Method", "Gamma", "Entropy_Threshold", "Generated_Text","Entropy_Plot"])
    torch.manual_seed(123)
    texts_as_large,_ = benchmark(autoregressive_sampling, "AS_large",tokenizer, large_model, num_tokens, top_k = top_k, top_p=top_p,start_idx=start_idx, end_idx=end_idx, enable_wandb=enable_wandb)

    torch.manual_seed(123)
    texts_as_small, _ = benchmark(autoregressive_sampling, "AS_small",tokenizer, small_model, num_tokens, top_k = top_k, top_p=top_p, start_idx=start_idx, end_idx=end_idx, enable_wandb=enable_wandb)

    torch.manual_seed(123)
    texts_sp,_ = benchmark(speculative_sampling, "SP",tokenizer, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, start_idx=start_idx, end_idx=end_idx,verbose=True, enable_wandb=enable_wandb)

    torch.manual_seed(123)
    texts_sp_entropy,entropies_list = benchmark(speculative_sampling_entropy_based, "SP_entropy",tokenizer, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, entropy_threshold = entropy_threshold, start_idx=start_idx, end_idx=end_idx,verbose=True, enable_wandb=enable_wandb)

    saved_files = [
        save_method_texts(output_dir, "AS_large", texts_as_large, start_idx),
        save_method_texts(output_dir, "AS_small", texts_as_small, start_idx),
        save_method_texts(output_dir, "SP", texts_sp, start_idx),
        save_method_texts(output_dir, "SP_entropy", texts_sp_entropy, start_idx),
    ]
    print(f"Generated texts saved to folder: {Path(output_dir).resolve()}")
    for p in saved_files:
        print(f"- {p}")

    # --- 将数据填充到表格 ---
    if enable_wandb and results_table is not None:
        for i in range(len(texts_as_large)):
            results_table.add_data(start_idx + i, "AS_Large", "N/A", "N/A", texts_as_large[i],None)
            if i < len(texts_as_small):
                results_table.add_data(start_idx + i, "AS_Small", "N/A", "N/A", texts_as_small[i],None)
            if i < len(texts_sp):
                results_table.add_data(start_idx + i, "SP", str(gamma), "N/A", texts_sp[i],None)
            if i < len(texts_sp_entropy):
                gen_text = texts_sp_entropy[i]
                current_entropies = entropies_list[i]
                fig_image = None
                if current_entropies:
                    # 重新获取 token 以便画图
                    token_ids = tokenizer.encode(gen_text, add_special_tokens=False)
                    tokens = [tokenizer.decode([tid]) for tid in token_ids]
                    
                    fig = visualize_token_entropy(
                        tokens=tokens, 
                        entropy_values=current_entropies, 
                        entropy_threshold=entropy_threshold,
                        title=f"Sample {start_idx + i} Entropy"
                    )
                    fig_image = wandb.Image(fig)
                    plt.close(fig) # 重要：防止内存溢出

                results_table.add_data(
                    start_idx + i, 
                    "SP_Entropy", 
                    str(gamma), 
                    str(entropy_threshold), 
                    gen_text, 
                    fig_image
                )

        # 最后 Log 表格到 WandB
        try:
            wandb.log({"Evaluation/Inference_Samples": results_table})
        except Exception as e:
            print(f"[WARN] wandb table log failed: {e}")

if __name__ == "__main__":
    args = parse_arguments()
    
    # 初始化 wandb
    enable_wandb = args.wandb_mode != "disabled"
    if enable_wandb:
        try:
            import wandb
            wandb.init(
                project="speculative-sampling-study",
                config=vars(args),  # 自动保存所有命令行参数
                name=f"run-{args.approx_model_name.split('/')[-1]}-{args.gamma}",
                mode=args.wandb_mode,
            )
        except Exception as e:
            print(f"[WARN] wandb init failed, continue without wandb: {e}")
            enable_wandb = False


    generate(
        args.approx_model_name,
        args.target_model_name,
        num_tokens=args.max_tokens,
        gamma=args.gamma,
        entropy_threshold=args.entropy_threshold,
        start_idx=args.start,
        end_idx=args.end,
        output_dir=args.output_dir,
        enable_wandb=enable_wandb,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    if enable_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"[WARN] wandb finish failed: {e}")