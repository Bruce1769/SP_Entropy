import torch
from .sampling_utils import sample
from .kvcache_model import KVCacheModel


def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)


@torch.inference_mode()
def speculative_sampling_entropy_based(
    prefix: torch.Tensor,
    approx_model_raw: torch.nn.Module,
    target_model_raw: torch.nn.Module,
    max_len: int,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    entropy_threshold: float = 2.5,
    eos_token_id: int = None,
    verbose: bool = False,
    relaxation_strategy: str = 'sqrt', # 'sqrt' or 'add_0.5'
    dynamic_gamma: bool = False,
    **kwargs
) -> tuple:
    approx_model = KVCacheModel(approx_model_raw, temperature, top_k, top_p)
    target_model = KVCacheModel(target_model_raw, temperature, top_k, top_p)

    all_entropies = []
    accepted_count, num_steps = 0, 0
    total_drafted = 0
    
    stats = {
        "total_evaluated": 0,
        "low_entropy_evaluated": 0,
        "high_entropy_evaluated": 0,
        "low_entropy_accepted": 0,
        "high_entropy_accepted": 0,
        "rescued_by_relaxation": 0
    }

    # Prefill
    q_approx = approx_model._forward_with_kvcache(prefix)  # (1, vocab)
    _ = target_model._forward_with_kvcache(prefix)

    next_token = sample(q_approx, temperature=1.0, top_k=0, top_p=0.0)
    prefix = torch.cat([prefix, next_token], dim=1)

    all_entropies.append(calculate_entropy(q_approx).item())

    T = prefix.shape[1] + max_len

    while prefix.shape[1] < T:
        num_steps += 1
        base_len = prefix.shape[1]

        # === A. Draft ===
        draft_tokens = []
        draft_probs = []
        for _ in range(gamma):
            q = approx_model._forward_with_kvcache(prefix[:, -1:])
            if q.dim() == 3:
                q = q.squeeze(1)
            token = sample(q, temperature=1.0, top_k=0, top_p=0.0)
            draft_tokens.append(token)
            draft_probs.append(q.squeeze(0))  # (vocab,)
            prefix = torch.cat([prefix, token], dim=1)
            
            if dynamic_gamma:
                draft_entropy = calculate_entropy(q).item()
                if draft_entropy > entropy_threshold:
                    break

        actual_gamma = len(draft_tokens)
        total_drafted += actual_gamma
        draft_seq = torch.cat(draft_tokens, dim=1)
        all_approx_probs = torch.stack(draft_probs, dim=0)  # (actual_gamma, vocab)

        # === B. Verify: single target forward with [next_token, draft_seq] ===
        verify_input = torch.cat([next_token, draft_seq], dim=1)  # (1, actual_gamma+1)
        target_out = target_model._forward_with_kvcache(verify_input)  # (1, actual_gamma+1, vocab)
        if target_out.dim() == 2:
            target_out = target_out.unsqueeze(0)
        all_target_probs = target_out.squeeze(0)  # (actual_gamma+1, vocab)

        # === C. Entropy-based acceptance ===
        n = 0
        step_entropies = []
        for i in range(actual_gamma):
            r = torch.rand(1, device=prefix.device)
            token_id = draft_tokens[i].squeeze().item()

            p = all_target_probs[i, token_id]
            q = all_approx_probs[i, token_id]

            draft_entropy = calculate_entropy(all_approx_probs[i].unsqueeze(0)).item()
            step_entropies.append(draft_entropy)
            
            stats["total_evaluated"] += 1

            if draft_entropy <= entropy_threshold:
                stats["low_entropy_evaluated"] += 1
                
                if relaxation_strategy == 'add_0.5':
                    accept_ratio = (p / q + 0.5).clamp(max=1.0)
                elif relaxation_strategy == 'aggressive':
                    accept_ratio = torch.tensor(1.0, device=p.device)
                else: # default is 'sqrt'
                    accept_ratio = torch.sqrt(p / q).clamp(max=1.0)
                    
                strict_ratio = (p / q).clamp(max=1.0)
                
                if r <= accept_ratio:
                    stats["low_entropy_accepted"] += 1
                    n += 1
                    if r > strict_ratio:
                        stats["rescued_by_relaxation"] += 1
                        
                    if eos_token_id is not None and token_id == eos_token_id:
                        approx_model.rollback(base_len + n)
                        target_model.rollback(base_len + n)
                        all_entropies.extend(step_entropies)
                        if kwargs.get('return_stats', False):
                            return prefix[:, :base_len + n], accepted_count + n, num_steps, all_entropies, stats
                        return prefix[:, :base_len + n], accepted_count + n, num_steps, all_entropies
                else:
                    break
            else:
                stats["high_entropy_evaluated"] += 1
                accept_ratio = (p / q).clamp(max=1.0)
                
                if r <= accept_ratio:
                    stats["high_entropy_accepted"] += 1
                    n += 1
                    if eos_token_id is not None and token_id == eos_token_id:
                        approx_model.rollback(base_len + n)
                        target_model.rollback(base_len + n)
                        all_entropies.extend(step_entropies)
                        if kwargs.get('return_stats', False):
                            return prefix[:, :base_len + n], accepted_count + n, num_steps, all_entropies, stats
                        return prefix[:, :base_len + n], accepted_count + n, num_steps, all_entropies
                else:
                    break

        # === D. Rollback & Resample ===
        approx_model.rollback(base_len + n)
        target_model.rollback(base_len + n)

        all_entropies.extend(step_entropies[:n])

        if n < actual_gamma:
            p_dist = all_target_probs[n]
            q_dist = all_approx_probs[n]
            diff_probs = torch.clamp(p_dist - q_dist, min=1e-9)
            diff_probs /= diff_probs.sum()
            next_token = torch.multinomial(diff_probs, num_samples=1).unsqueeze(0)
        else:
            bonus_logits = all_target_probs[-1, :]
            next_token = sample(bonus_logits.unsqueeze(0), temperature=1.0, top_k=0, top_p=0.0)

        prefix = torch.cat([prefix[:, :base_len + n], next_token], dim=1)
        accepted_count += n

        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    if verbose:
        accept_ratio = accepted_count / total_drafted if total_drafted > 0 else 0.0
        print(f"SP_entropy accept_ratio: {accept_ratio:.4f} accepted: {accepted_count} steps: {num_steps} avg_gamma: {total_drafted/num_steps:.2f}")
    
    if kwargs.get('return_stats', False):
        return prefix, accepted_count, num_steps, all_entropies, stats
    return prefix, accepted_count, num_steps, all_entropies
