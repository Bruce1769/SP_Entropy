import torch
from sampling.sampling_utils import norm_logits, sample

@torch.no_grad()
def entropy_selective_sampling(
    x: torch.Tensor, 
    small_model: torch.nn.Module, 
    large_model: torch.nn.Module, 
    N: int, 
    entropy_threshold: float = 0.1,
    temperature: float = 1.0, 
    top_k: int = 0, 
    top_p: float = 0.0,
    eos_token_id: int = None
):
    """
    Experiment 2 Baseline: Entropy-Selective Generation
    - Use small model to generate.
    - If small model's entropy > threshold, fallback to large model for this token.
    - KV caches are maintained for both models.
    """
    n = len(x[0])
    T = n + N
    
    small_past_key_values = None
    large_past_key_values = None
    
    generated_tokens = []
    large_model_calls = 0
    total_steps = 0

    while n < T:
        total_steps += 1
        
        # 1. Forward small model
        if small_past_key_values:
            last_ids = x[:, -1:]
            small_outputs = small_model(last_ids, past_key_values=small_past_key_values, use_cache=True)
        else:
            small_outputs = small_model(x)
            
        small_logits = small_outputs.logits[:, -1, :]
        small_probs = norm_logits(small_logits, temperature, top_k, top_p)
        
        # Calculate entropy
        ent = -torch.sum(small_probs * torch.log(small_probs + 1e-10), dim=-1).item()
        
        # Always forward large model to keep cache in sync (for simplicity in this baseline)
        if large_past_key_values:
            large_outputs = large_model(last_ids, past_key_values=large_past_key_values, use_cache=True)
        else:
            large_outputs = large_model(x)
            
        # 2. Decide which model to trust based on entropy
        if ent > entropy_threshold:
            # High entropy -> Small model is uncertain -> Trust Large model
            large_model_calls += 1
            large_logits = large_outputs.logits[:, -1, :]
            large_probs = norm_logits(large_logits, temperature, top_k, top_p)
            idx_next = sample(large_probs, temperature, top_k, top_p)
        else:
            # Low entropy -> Small model is confident -> Trust Small model
            idx_next = sample(small_probs, temperature, top_k, top_p)
            
        # Update caches
        small_past_key_values = small_outputs.past_key_values
        large_past_key_values = large_outputs.past_key_values
        
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            break
            
        x = torch.cat((x, idx_next), dim=1)
        generated_tokens.append(idx_next.item())
        n += 1
        
    return x, large_model_calls, total_steps
