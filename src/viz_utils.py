import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_attention_distribution(attentions, layer_idx=0, head_idx=0):
    
    attn = attentions[layer_idx]
    if attn.dim() == 4:
        attn = attn.mean(dim=0)
    
    cls_attention = attn[head_idx, 0, 1:].cpu().detach().numpy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(cls_attention, color='#2E86AB', linewidth=2.5, alpha=0.8, marker='o', markersize=4)
    plt.fill_between(range(len(cls_attention)), cls_attention, alpha=0.3, color='#2E86AB')
    
    top_5_indices = np.argsort(cls_attention)[-5:][::-1]
    top_5_weights = cls_attention[top_5_indices]
    
    for idx, weight in zip(top_5_indices, top_5_weights):
        plt.plot(idx, weight, 'ro', markersize=8)
        plt.annotate(f'Срез {idx}\n{weight:.3f}', 
                    (idx, weight), xytext=(10, 15), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.1),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.xlabel('Номер среза', fontsize=12, fontweight='bold')
    plt.ylabel('Вес внимания', fontsize=12, fontweight='bold')
    plt.title(f'Распределение внимания по срезам\n(Слой {layer_idx}, Голова {head_idx})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return top_5_indices, top_5_weights

def plot_top_slices(x, attentions, top_indices, layer_idx=0, head_idx=0):
    
    B, N, C, H, W = x.shape
    x_vis = x[0].cpu()
    
    attn = attentions[layer_idx]
    if attn.dim() == 4:
        attn = attn.mean(dim=0)
    cls_attention = attn[head_idx, 0, 1:].cpu().detach().numpy()
    
    n_cols = min(4, len(top_indices))
    n_rows = (len(top_indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, slice_idx in enumerate(top_indices):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        slice_img = x_vis[slice_idx].permute(1, 2, 0).numpy()
        
        if slice_img.shape[2] == 3:
            slice_img = np.mean(slice_img, axis=2)
        elif slice_img.shape[2] == 1:
            slice_img = slice_img[:, :, 0]
        
        vmin, vmax = np.percentile(slice_img, [5, 95])
        slice_img = np.clip(slice_img, vmin, vmax)
        slice_img = (slice_img - vmin) / (vmax - vmin + 1e-8)
        
        im = ax.imshow(slice_img, cmap='gray', aspect='equal')
        weight = cls_attention[slice_idx]
        
        ax.set_title(f'Срез {slice_idx}\nВнимание: {weight:.3f}', 
                    fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        color_intensity = weight / cls_attention.max()
        border_color = plt.cm.Reds(color_intensity)
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(3)
    
    plt.suptitle(f'Срезы с максимальным вниманием\n(Слой {layer_idx}, Голова {head_idx})', 
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()

def plot_attention_heatmap(attentions, layer_idx=0):

    attn = attentions[layer_idx]
    if attn.dim() == 4:
        attn = attn.mean(dim=0)
    
    num_heads, seq_len, _ = attn.shape
    num_slices = seq_len - 1
    
    cls_attention_all_heads = attn[:, 0, 1:].cpu().detach().numpy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    im1 = ax1.imshow(cls_attention_all_heads, cmap='YlOrRd', aspect='auto', 
                    interpolation='nearest')
    ax1.set_xlabel('Номер среза', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Номер головы', fontsize=12, fontweight='bold')
    ax1.set_title(f'Тепловая карта внимания по головам (Слой {layer_idx})', 
                 fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(np.arange(0, num_slices, max(1, num_slices//10)))
    ax1.set_yticks(range(num_heads))
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Вес внимания', fontsize=11)
    
    avg_attention = cls_attention_all_heads.mean(axis=0)
    
    colors = ['#FF6B6B' if i in np.argsort(avg_attention)[-5:] else '#4ECDC4' 
             for i in range(len(avg_attention))]
    
    ax2.bar(range(len(avg_attention)), avg_attention, color=colors, alpha=0.8)
    ax2.set_xlabel('Номер среза', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Средний вес внимания', fontsize=12, fontweight='bold')
    ax2.set_title('Усредненное внимание по всем головам', 
                 fontsize=14, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')
    
    top_5_indices = np.argsort(avg_attention)[-5:][::-1]
    for idx in top_5_indices:
        ax2.text(idx, avg_attention[idx] + 0.01, f'#{idx}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='red')
    
    plt.tight_layout()
    plt.show()
    
    return avg_attention, top_5_indices

def plot_attention_statistics(attentions):
    
    num_layers = len(attentions)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    layer_stats = []
    for layer_idx, attn in enumerate(attentions):
        if attn.dim() == 4:
            attn = attn.mean(dim=0).mean(dim=0)
        
        cls_attn = attn[0, 1:].cpu().detach().numpy()
        
        layer_stats.append({
            'layer': layer_idx,
            'max': cls_attn.max(),
            'min': cls_attn.min(), 
            'mean': cls_attn.mean(),
            'std': cls_attn.std(),
            'top_slice': np.argmax(cls_attn),
            'top_value': cls_attn.max()
        })
    
    layers = [s['layer'] for s in layer_stats]
    max_vals = [s['max'] for s in layer_stats]
    
    axes[0].bar(layers, max_vals, color='skyblue', alpha=0.8)
    axes[0].set_xlabel('Номер слоя', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Максимальное внимание', fontsize=12, fontweight='bold')
    axes[0].set_title('Максимальное внимание по слоям', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    mean_vals = [s['mean'] for s in layer_stats]
    axes[1].bar(layers, mean_vals, color='lightcoral', alpha=0.8)
    axes[1].set_xlabel('Номер слоя', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Среднее внимание', fontsize=12, fontweight='bold')
    axes[1].set_title('Среднее внимание по слоям', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    top_slices = [s['top_slice'] for s in layer_stats]
    axes[2].bar(layers, top_slices, color='lightgreen', alpha=0.8)
    axes[2].set_xlabel('Номер слоя', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Номер среза', fontsize=12, fontweight='bold')
    axes[2].set_title('Срезы с максимальным вниманием по слоям', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    std_vals = [s['std'] for s in layer_stats]
    axes[3].bar(layers, std_vals, color='gold', alpha=0.8)
    axes[3].set_xlabel('Номер слоя', fontsize=12, fontweight='bold')
    axes[3].set_ylabel('Стандартное отклонение', fontsize=12, fontweight='bold')
    axes[3].set_title('Разброс внимания по слоям', fontsize=14, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nСТАТИСТИКА ПО СЛОЯМ:")
    print("=" * 50)
    for stats in layer_stats:
        print(f"Слой {stats['layer']}: "
              f"max={stats['max']:.3f}, mean={stats['mean']:.3f}, "
              f"std={stats['std']:.3f}, top_slice={stats['top_slice']}")
    
    return layer_stats

def plot_slice_comparison(x, slice_indices, title="Сравнение срезов"):
    
    x_vis = x[0].cpu()
    n_slices = len(slice_indices)
    
    fig, axes = plt.subplots(1, n_slices, figsize=(4*n_slices, 4))
    if n_slices == 1:
        axes = [axes]
    
    for i, slice_idx in enumerate(slice_indices):
        slice_img = x_vis[slice_idx].permute(1, 2, 0).numpy()
        
        if slice_img.shape[2] == 3:
            slice_img = np.mean(slice_img, axis=2)
        elif slice_img.shape[2] == 1:
            slice_img = slice_img[:, :, 0]
        
        vmin, vmax = np.percentile(slice_img, [2, 98])
        slice_img_display = np.clip(slice_img, vmin, vmax)
        slice_img_display = (slice_img_display - vmin) / (vmax - vmin + 1e-8)
        
        axes[i].imshow(slice_img_display, cmap='gray')
        axes[i].set_title(f'Срез {slice_idx}', fontsize=14, fontweight='bold')
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        
        axes[i].text(0.05, 0.95, f'Min: {slice_img.min():.2f}\nMax: {slice_img.max():.2f}', 
                    transform=axes[i].transAxes, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                    verticalalignment='top')
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.show()


def get_most_important_slices(attentions, x, top_k=8):
    
    all_attentions = []
    
    for layer_idx, attn in enumerate(attentions):
        if attn.dim() == 4:
            attn = attn.mean(dim=0)
            
        layer_heads_attention = attn[:, 0, 1:].cpu().detach().numpy()
        all_attentions.append(layer_heads_attention)
    
    combined_attention = np.concatenate(all_attentions, axis=0)
    
    overall_importance = combined_attention.mean(axis=0)
    
    top_indices = np.argsort(overall_importance)[-top_k:][::-1]
    top_scores = overall_importance[top_indices]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(overall_importance, alpha=0.7)
    plt.xlabel('Номер среза')
    plt.ylabel('Общая важность')
    plt.title('Общая важность срезов (все слои + головы)')
    plt.grid(True, alpha=0.3)
    
    for idx, score in zip(top_indices, top_scores):
        plt.plot(idx, score, 'ro', markersize=6)
        plt.annotate(f'#{idx}', (idx, score), xytext=(5, 5), 
                    textcoords='offset points', fontweight='bold')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(top_scores)), top_scores, color='lightcoral', alpha=0.7)
    plt.xticks(range(len(top_scores)), [f'Срез {idx}' for idx in top_indices], rotation=45)
    plt.ylabel('Важность')
    plt.title(f'Топ-{top_k} самых важных срезов')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Самые важные срезы (по всем слоям и головам):")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"  {i+1}. Срез {idx} - важность: {score:.4f}")
    
    plot_simple_slices(x, top_indices, "Самые важные срезы")
    
    return top_indices, top_scores

def plot_simple_slices(x, slice_indices, title):
    x_vis = x[0].cpu()
    n_slices = len(slice_indices)
    
    fig, axes = plt.subplots(2, n_slices // 2, figsize=(10, 10))
    if n_slices == 1:
        axes = [axes]
    
    for i, slice_idx in enumerate(slice_indices):
        slice_img = x_vis[slice_idx].permute(1, 2, 0).numpy()
        
        if slice_img.shape[2] == 3:
            slice_img = np.mean(slice_img, axis=2)
        
        axes[i // 4][i % 4].imshow(slice_img, cmap='gray')
        axes[i // 4][i % 4].set_title(f'Срез {slice_idx}')
        axes[i // 4][i % 4].set_xticks([])
        axes[i // 4][i % 4].set_yticks([])
    
    plt.suptitle(title, fontweight='bold')
    plt.tight_layout()
    plt.show()