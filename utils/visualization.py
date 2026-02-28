import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.colors import Normalize

def visualize_token_entropy(
    tokens: list[str], 
    entropy_values: list[float], 
    entropy_threshold: float = None, 
    title: str = "Token Entropy Visualization", 
    save_path: str = None
) -> matplotlib.figure.Figure:
    """
    可视化 Token 的熵值分布。
    
    参数:
    - tokens: list of str, 解码后的 token 列表
    - entropy_values: list of float, 每个 token 对应的熵值
    - entropy_threshold: float, 可选，熵拒绝的阈值，将在色标上标出
    - title: str, 图表标题
    - save_path: str, 可选，本地保存路径
    
    返回:
    - fig: matplotlib.figure.Figure 对象，可直接传给 wandb.Image(fig)
    """
    if not entropy_values or not tokens:
        return None
        
    # 1. 预处理数据：确保长度一致并处理 None 值
    safe_entropies = [float(e) if e is not None else 0.0 for e in entropy_values]
    if len(tokens) > len(safe_entropies):
        tokens = tokens[:len(safe_entropies)]
    
    # 2. 设置颜色映射 (从深蓝到深红)
    cmap = cm.get_cmap('coolwarm')
    min_entropy = np.min(safe_entropies)
    max_entropy = np.max(safe_entropies)
    
    # 增加一个小 buffer 避免除零，并确保范围能包含阈值线
    display_min = min(min_entropy, entropy_threshold if entropy_threshold else min_entropy)
    display_max = max(max_entropy, entropy_threshold if entropy_threshold else max_entropy)
    if display_min == display_max:
        display_min -= 0.1
        display_max += 0.1
        
    norm = Normalize(vmin=display_min, vmax=display_max)
    colors = [cmap(norm(e)) for e in safe_entropies]
    
    # 3. 布局参数
    font_size = 12
    X_SPACING = 0.005      # Token 块之间的水平间距
    char_width_unit = 0.32 # 每个字符占用的宽度单位 (针对 monospace 字体)
    BOX_PAD = 0.2          # Token 块内部的边距
    MAX_PLOT_WIDTH = 20    # 每行的最大宽度
    FIXED_FIG_WIDTH = 15   # 画布固定宽度
    LINE_HEIGHT = 0.7      # 换行时的行高

    # 创建画布
    fig, ax = plt.subplots(figsize=(FIXED_FIG_WIDTH, 4))
    ax.axis('off')
    
    x_pos = X_SPACING
    vertical_center = 0.95
    current_line_width = 0
    line_count = 1

    # 4. 逐个绘制 Token 块
    for token, color in zip(tokens, colors):
        # 处理特殊字符防止 LaTeX 报错或渲染问题
        display_token = (token.replace('\n', '¶')
                              .replace(' ', '·')
                              .replace('$', '\\$')
                              .replace('_', '\\_'))
        
        # 动态计算该 Token 块的宽度
        token_body_width = len(display_token) * char_width_unit * font_size / 40 + 2 * BOX_PAD
        token_block_width = token_body_width + X_SPACING
        
        # 自动换行检查
        if current_line_width + token_block_width > MAX_PLOT_WIDTH:
            line_count += 1
            vertical_center -= LINE_HEIGHT
            x_pos = X_SPACING
            current_line_width = 0

        # 绘制带背景色的 Token 块
        ax.text(x_pos, vertical_center, display_token, fontsize=font_size,
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', boxstyle=f'round,pad={BOX_PAD}'),
                verticalalignment='center', horizontalalignment='left', family='monospace')
        
        x_pos += token_block_width
        current_line_width += token_block_width

    # 5. 设置坐标范围和动态高度
    ax.set_ylim(vertical_center - LINE_HEIGHT, 1.1)
    ax.set_xlim(0, MAX_PLOT_WIDTH)
    
    final_height = max(4, line_count * LINE_HEIGHT + 2.5)
    fig.set_size_inches(FIXED_FIG_WIDTH, final_height)
    
    # 6. 添加颜色条 (Colorbar) 和阈值参考线
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(safe_entropies)
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', fraction=0.04, pad=0.15)
    cbar.set_label('Token Entropy (Blue=Confident/Draft, Red=Uncertain/Target)')
    
    if entropy_threshold is not None:
        # 在色标上画出阈值虚线
        cbar.ax.axvline(entropy_threshold, color='white', linestyle='--', linewidth=2)
        # 在虚线上方标注文字
        cbar.ax.text(entropy_threshold, 1.2, f'Reject Threshold: {entropy_threshold}', 
                     color='red', fontweight='bold', ha='center', 
                     transform=cbar.ax.get_xaxis_transform())

    plt.title(title, fontsize=14, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.tight_layout()
    return fig