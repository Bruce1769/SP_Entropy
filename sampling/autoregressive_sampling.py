import torch

from tqdm import tqdm
from sampling.sampling_utils import norm_logits, sample

@torch.no_grad()
def autoregressive_sampling(x : torch.Tensor, model : torch.nn.Module, N : int, 
                            temperature : float = 1.0, top_k : int = 0, top_p : float = 0,
                            eos_token_id : int = None):
    n = len(x)
    T = len(x) + N

    past_key_values = None
    while n < T:
        # outputs = model(x)
        if past_key_values:
            last_ids = x[:, -1]
            if last_ids.dim() == 1:
                last_ids = torch.unsqueeze(last_ids, 0)
            outputs = model(last_ids, past_key_values = past_key_values, use_cache = True)
        else:
            outputs = model(x)
        last_p = norm_logits(outputs.logits[::, -1, :], temperature, top_k, top_p)
        past_key_values = outputs.past_key_values
        idx_next = sample(last_p,temperature,top_k,top_p)

        # === 2. 增加停止条件检查 ===
        if eos_token_id is not None and (idx_next == eos_token_id).all():
            # 如果生成的 token 是结束符，则停止生成
            # 注意：如果 batch_size > 1，这里需要更复杂的逻辑（比如记录每个样本是否完成）
            # 对于 batch_size = 1，直接 break 即可
            break
        # =========================

        x = torch.cat((x, idx_next), dim=1)
        n += 1
    return x

