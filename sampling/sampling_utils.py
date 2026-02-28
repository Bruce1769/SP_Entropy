import torch
from torch.nn import functional as F
import pdb
from IPython import embed

# copy from https://github.com/LeeSinLiang/microGPT/blob/ed40cf9780dbeb180adfe94c227d4aa97e69250e/gpt.py
def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0):
    """

    Args:
        logits (torch.Tensorpe_): 2D tensor with shape (batch, vocab)
        top_k (int, optional): top_k. Defaults to 0.
        top_p (float, optional): top_p. Defaults to 0.0.

    Returns:
        torch.Tensor: a renormalized logits
    """
    if top_k > 0:
        filter = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < filter[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)
        filter = cumulative_probs > top_p
        filter[..., 1:] = filter[..., :-1].clone()
        filter[..., 0] = 0
        indices_to_remove = filter.scatter(1, sorted_indices, filter)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(logits : torch.Tensor, temperature : float, top_k : float, top_p : float) -> torch.Tensor:
    """

    Args:
        logits (torch.Tensor): shape (1, vocab)
        temperature (float): temperature
        top_k (float): top_k
        top_p (float): top_p

    Returns:
        torch.Tensor: next token with shape as (batch,  1)
    """
    assert logits.dim() == 2
    logits = logits / temperature
    logits = top_k_top_p_filter(logits, top_k=top_k, top_p=top_p)
    probs = F.softmax(logits, dim=1)
    return probs


def sample(probs: torch.Tensor, 
           temperature: float = 1.0,
           top_k: int = 50, 
           top_p: float = 0.9
           ):
    """
    Args:
        probs (torch.Tensor): 概率分布 (Input should be Probabilities, not Logits).
        top_k (int): 保留前 k 个 token.
        top_p (float): 核采样阈值.
        temperature (float): 温度系数. 
                             T < 1 使分布更尖锐(更保守), 
                             T > 1 使分布更平缓(更随机).
    """
    # 1. 保护原始数据
    probs = probs.clone()

    # pdb.set_trace()

    # 2. 应用温度 (Temperature)
    # 注意：对于概率值 P，应用温度 T 等价于 P^(1/T) 然后重新归一化
    if temperature != 1.0:
        if temperature <= 1e-5:
            # 极低温度处理为贪婪采样 (Greedy): 将最大概率置为1，其余为0
            _, max_indices = torch.max(probs, dim=-1, keepdim=True)
            probs.zero_()
            probs.scatter_(-1, max_indices, 1.0)
        else:
            # 标准温度处理
            # 加上 1e-10 防止 0 的幂运算出问题（虽然理论上 0^x = 0）
            probs = torch.pow(probs, 1.0 / temperature)
            # 重新归一化
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    # 3. Top-K 过滤
    if top_k is not None and 0 < top_k < probs.size(-1):
        # print(f"top_k: {top_k},top_k type: {type(top_k)}")
        val, _ = torch.topk(probs, top_k)
        min_val = val[..., -1, None]
        probs[probs < min_val] = 0
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    # 4. Top-P (Nucleus) 过滤
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 移除累积概率超过 top_p 的部分（保留刚超过的那一个）
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # 映射回原始索引
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        
        probs[indices_to_remove] = 0
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

    # 5. 采样
    # 此时 probs 已经经过了 Temp -> TopK -> TopP 的层层筛选和归一化
    idx_next = torch.multinomial(probs, num_samples=1)

    return idx_next


def max_fn(x):
    """
        norm(max (x, 0))
    """
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=1, keepdim=True) 
    return x_max / x_max_sum

def calculate_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    使用 torch.special.entr 计算熵 (单位: nats/自然单位)
    输入:
        probs: 概率分布 Tensor，所有元素应 >= 0 且在 dim 维度和为 1
        dim: 计算熵的维度，默认为最后一个维度
    输出:
        熵值 Tensor
    """
    return torch.sum(torch.special.entr(probs), dim=dim)

def format_math_prompt(problem_text):
    prompt = [
        {
            "role": "system", 
            "content": "You are a highly proficient mathematician. Solve the problem step-by-step, providing clear explanations. Always enclose your final answer within \\boxed{}."
        },
        {
            "role": "user", 
            "content": problem_text
        }
    ]
    return prompt

def visualize_token_entropy(tokens, entropy_values, title="Token Entropy Visualization", save_path=None):
    """可视化token及其熵值"""
    if not entropy_values:
        return

    safe_entropies = [e if e is not None else 0.0 for e in entropy_values]
    
    cmap = plt.get_cmap('coolwarm')
    if len(safe_entropies) > 0:
        min_entropy = np.min(safe_entropies)
        max_entropy = np.max(safe_entropies)
    else:
        min_entropy, max_entropy = 0, 1
    
    if min_entropy == max_entropy:
        min_entropy -= 0.1
        max_entropy += 0.1
      
    norm = Normalize(vmin=min_entropy, vmax=max_entropy)
    colors = [cmap(norm(e)) for e in safe_entropies]
    
    font_size = 14
    X_SPACING = 0.005 
    char_width_unit = 0.35 
    BOX_PAD = 0.3
    MAX_PLOT_WIDTH = 20  
    FIXED_FIG_WIDTH = 15 
    LINE_HEIGHT = 0.5  
    
    fig, ax = plt.subplots(figsize=(FIXED_FIG_WIDTH, 3)) 
    ax.axis('off')
    
    x_pos = X_SPACING 
    vertical_center = 0.8 
    current_line_width = 0
    line_count = 1

    for token, color, entropy in zip(tokens, colors, safe_entropies):
        display_token = token.replace('\n', '¶').replace('$', '\\$')
        token_body_width = len(display_token) * char_width_unit * font_size / 50 + 2 * BOX_PAD
        token_block_width = token_body_width + X_SPACING
        
        if current_line_width + token_block_width > MAX_PLOT_WIDTH:
            line_count += 1
            vertical_center -= LINE_HEIGHT
            x_pos = X_SPACING 
            current_line_width = 0

        ax.text(
            x_pos, vertical_center, 
            display_token,
            fontsize=font_size,
            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle=f'round,pad={BOX_PAD}'),
            verticalalignment='center',
            horizontalalignment='left'
        )
        
        x_pos += token_block_width
        current_line_width += token_block_width

    ax.set_ylim(vertical_center - LINE_HEIGHT/2, 1) 
    ax.set_xlim(0, MAX_PLOT_WIDTH + X_SPACING) 
    final_height = max(5, line_count * LINE_HEIGHT + 2)
    fig.set_size_inches(FIXED_FIG_WIDTH, final_height, forward=True)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(safe_entropies) 
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.08)
    cbar.set_label('Token Entropy (Blue=Low/Confident, Red=High/Uncertain)')
    
    mean_entropy = np.mean(safe_entropies)
    cbar.set_ticks([min_entropy, mean_entropy, max_entropy])
    cbar.ax.set_xticklabels([
        f'Min: {min_entropy:.2f}', 
        f'Avg: {mean_entropy:.2f}', 
        f'Max: {max_entropy:.2f}'
    ])
    
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)