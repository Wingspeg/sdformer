import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, ArrowStyle

# 设置绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(20, 10))
ax.set_xlim(0, 130)  # 进一步扩大x轴范围
ax.set_ylim(0, 100)
ax.axis('off')
ax.set_title("SDFormer Architecture: Scale-aware Dynamic Forecaster", fontsize=16, fontweight='bold', pad=20)

# --- 1. 定义颜色 ---
color_fine = '#FF7F50'   # 珊瑚红 (细粒度/高频)
color_mid = '#9370DB'    # 中紫 (中粒度)
color_coarse = '#4682B4' # 钢蓝 (粗粒度/低频)
color_gate = '#FFD700'   # 金色 (门控)
color_text = '#333333'

# --- 2. 绘制输入层 (Input Layer) ---
# 输入框的位置和大小
input_box_x = 5
input_box_y = 35
input_box_w = 10
input_box_h = 30

# 绘制输入框
ax.add_patch(patches.Rectangle((input_box_x, input_box_y), input_box_w, input_box_h, 
                                linewidth=2, edgecolor='black', facecolor='#f0f0f0', linestyle='-'))

# 标题文字（放在框上方，完全在框外）
ax.text(input_box_x + input_box_w/2, input_box_y + input_box_h + 8, 
        "Input Sequence ($X_t$)", ha='center', fontsize=12, fontweight='bold')
ax.text(input_box_x + input_box_w/2, input_box_y + input_box_h + 5, 
        "(Normal + Abnormal Peak)", ha='center', fontsize=9, style='italic')

# 在框内画波形，确保不超出边界
# 使用参数化坐标，留出边距
margin = 0.5
t_simple = np.linspace(0, 1, 60)
x_wave = input_box_x + margin + t_simple * (input_box_w - 2*margin)

# 创建带尖峰的波形，归一化到框内
y_wave_base = np.sin(t_simple * 2 * np.pi) * 0.3
# 在 t=0.6 位置添加尖峰
peak_pos = 0.6
y_wave_peak = np.exp(-((t_simple - peak_pos)**2) / 0.005) * 0.6
y_wave = y_wave_base + y_wave_peak

# 映射到框内的 y 坐标（居中，留出上下边距）
y_center = input_box_y + input_box_h/2
y_scale = (input_box_h - 4) / 2  # 留出上下各2单位的边距
y_coords = y_center + y_wave * y_scale

# 绘制波形
ax.plot(x_wave, y_coords, color='black', linewidth=2)

# 标注尖峰（箭头指向峰值位置）
peak_x = input_box_x + margin + peak_pos * (input_box_w - 2*margin)
peak_y = y_center + (y_wave_base[int(peak_pos*60)] + y_wave_peak[int(peak_pos*60)]) * y_scale
ax.annotate("Abnormal\nPeak", xy=(peak_x, peak_y), xytext=(peak_x + 3, peak_y + 8),
            arrowprops=dict(arrowstyle="->", color='red', lw=2), 
            color='red', fontsize=9, fontweight='bold', ha='left')

# --- 2.5 异常感知归一化 (Anomaly-Aware Normalization) ---
norm_x = 20
norm_y = 50
norm_w = 8
norm_h = 20
color_norm = '#32CD32'  # 绿色

# 从输入到归一化的箭头
ax.annotate("", xy=(norm_x, norm_y), xytext=(input_box_x + input_box_w, 50),
            arrowprops=dict(arrowstyle="->", color=color_norm, lw=2, mutation_scale=20))

# 归一化模块框
norm_rect = patches.FancyBboxPatch((norm_x, norm_y - norm_h/2), norm_w, norm_h, 
                                   boxstyle="round,pad=0.15", linewidth=2.5, 
                                   edgecolor=color_norm, facecolor='#f0fff0')
ax.add_patch(norm_rect)

# 标题（放在框上方）
ax.text(norm_x + norm_w/2, norm_y + norm_h/2 + 3, "Anomaly-Aware", ha='center', fontsize=11, fontweight='bold', color=color_norm)
ax.text(norm_x + norm_w/2, norm_y + norm_h/2 + 0.5, "Normalization", ha='center', fontsize=11, fontweight='bold', color=color_norm)

# 公式（框内中部）- 使用 Kurtosis 和 SpikeRatio
ax.text(norm_x + norm_w/2, norm_y + 4, r"Kurtosis($X$)", ha='center', fontsize=9, family='serif')
ax.text(norm_x + norm_w/2, norm_y + 1.5, r"SpikeRatio($X$)", ha='center', fontsize=9, family='serif')

# 说明文字（框内下部）
ax.text(norm_x + norm_w/2, norm_y - 2, r"Adaptive $\alpha$ gate", ha='center', fontsize=9, style='italic')
ax.text(norm_x + norm_w/2, norm_y - 4.5, r"median $\leftrightarrow$ mean", ha='center', fontsize=9, style='italic')

# --- 3. 绘制多分辨率解耦分支 (Decoupling Branches) ---
branches = [
    {"name": "Fine-grained", "color": color_fine, "y_pos": 80, "label": "High-Freq Detector", "win_w": 12},
    {"name": "Mid-grained", "color": color_mid, "y_pos": 50, "label": "Transition Bridge", "win_w": 12},
    {"name": "Coarse-grained", "color": color_coarse, "y_pos": 20, "label": "Low-Freq Trend", "win_w": 12}
]

start_x = 35  # 增加起始位置，给归一化模块更多空间
block_w = 15
block_h = 12
window_encoder_gap = 5  # 滑动窗口和Encoder之间的间隔

# 绘制从输入到分支的箭头
for b in branches:
    # 箭头 - 从归一化模块指向各分支
    ax.annotate("", xy=(start_x, b["y_pos"]), xytext=(norm_x + norm_w, norm_y),
                arrowprops=dict(arrowstyle="->", color=b["color"], lw=2, mutation_scale=20))
    
    # 分支标题
    ax.text(start_x + block_w/2, b["y_pos"] + 14, b["name"], ha='center', fontsize=11, fontweight='bold', color=b["color"])
    ax.text(start_x + block_w/2, b["y_pos"] + 11, b["label"], ha='center', fontsize=9, style='italic')

    # 1. 滑动窗口示意 (不同宽度)
    rect_win = patches.Rectangle((start_x, b["y_pos"] - 2), b["win_w"], 4, linewidth=1.5, edgecolor=b["color"], facecolor='white', linestyle='--')
    ax.add_patch(rect_win)
    
    # 窗口内的波形示意 (细粒度保留尖峰，粗粒度平滑)
    # 使用参数化的 t 来确保波形在窗口内
    t_wave = np.linspace(0, 1, 30)
    wx = start_x + t_wave * b["win_w"]
    
    if b["name"].startswith("Fine"):
        # 画个尖峰
        wy = b["y_pos"] + np.exp(-((t_wave - 0.5)**2)/0.02) * 1.8
        ax.plot(wx, wy, color=b["color"], linewidth=2)
    elif b["name"].startswith("Coarse"):
        # 画个平滑线
        wy = b["y_pos"] + np.sin(t_wave * 2 * np.pi) * 0.4
        ax.plot(wx, wy, color=b["color"], linewidth=2)
    else:
        # 中等
        wy = b["y_pos"] + np.sin(t_wave * 4 * np.pi) * 0.8 + np.exp(-((t_wave - 0.5)**2)/0.05) * 0.8
        ax.plot(wx, wy, color=b["color"], linewidth=2)

    # 从滑动窗口到Encoder的箭头
    window_end_x = start_x + b["win_w"]
    enc_x = window_end_x + window_encoder_gap
    ax.annotate("", xy=(enc_x, b["y_pos"]), xytext=(window_end_x, b["y_pos"]),
                arrowprops=dict(arrowstyle="->", color=b["color"], lw=2, mutation_scale=15))

    # 2. Encoder Block
    enc_rect = patches.FancyBboxPatch((enc_x, b["y_pos"] - 4), block_w, block_h, boxstyle="round,pad=0.1", linewidth=2, edgecolor=b["color"], facecolor='white')
    ax.add_patch(enc_rect)
    ax.text(enc_x + block_w/2, b["y_pos"] + 2, "Transformer\nEncoder", ha='center', fontsize=10)

    # 3. 显著性重校准机制 (Saliency Recalibration) - 核心创新点
    # 在 Encoder 后画一个热力图条
    sal_x = enc_x + block_w + 2
    sal_w = 6
    sal_h = 4
    sal_rect = patches.Rectangle((sal_x, b["y_pos"] - 2), sal_w, sal_h, linewidth=1.5, edgecolor='black', facecolor='#fff5e6')
    ax.add_patch(sal_rect)
    ax.text(sal_x + sal_w/2, b["y_pos"] + 3, "ATSR", ha='center', fontsize=9, fontweight='bold')
    
    # 根据粒度设置高亮块的数量
    if b["name"].startswith("Fine"):
        highlight_indices = [2]  # 细粒度：中间1个高亮
    elif b["name"].startswith("Mid"):
        highlight_indices = [1, 2]  # 中粒度：中间2个高亮
    else:  # Coarse
        highlight_indices = [0, 1, 2, 3]  # 粗粒度：前4个高亮
    
    # 绘制5个块（背景色 + 高亮色）
    show_amplify = False
    for i in range(5):
        if i in highlight_indices:
            h_intensity = 0.8  # 高亮块
            if b["name"].startswith("Fine") and i == 2:
                h_intensity = 1.0  # Fine-grained 的高亮块最亮
                show_amplify = True
        else:
            h_intensity = 0.3  # 背景块（浅色）
            
        color_val = (1, 0, 0, h_intensity)  # Red with alpha
        ax.add_patch(patches.Rectangle((sal_x + i*1.2, b["y_pos"] - 2), 1, sal_h, 
                                       facecolor=color_val, edgecolor='none'))
    
    # 只在 Fine-grained 的高亮块显示 "Amplify!"
    if show_amplify:
        ax.text(sal_x + 2*1.2 + 0.5, b["y_pos"] - 4, "Amplify!", 
               ha='center', fontsize=8, color='red', fontweight='bold')

# --- 4. 自适应门控融合 (Adaptive Gating Fusion) ---
fuse_x = 85  # 调整融合模块的位置，适应新的布局
fuse_y = 50
fuse_w = 20
fuse_h = 25

# 汇聚箭头 - 从每个分支的 Saliency Recalibration 模块指向融合模块
for b in branches:
    # 计算 Saliency Recalibration 的结束位置
    window_end_x = start_x + b["win_w"]
    enc_x = window_end_x + window_encoder_gap
    sal_x = enc_x + block_w + 2
    sal_w = 6
    end_branch_x = sal_x + sal_w
    end_branch_y = b["y_pos"]
    ax.annotate("", xy=(fuse_x, fuse_y), xytext=(end_branch_x, end_branch_y),
                arrowprops=dict(arrowstyle="->", color='gray', lw=1.5, linestyle='--'))

# 融合模块框
fuse_rect = patches.FancyBboxPatch((fuse_x, fuse_y - fuse_h/2), fuse_w, fuse_h, boxstyle="round4,pad=0.2", linewidth=3, edgecolor=color_gate, facecolor='#fffdf0')
ax.add_patch(fuse_rect)
ax.text(fuse_x + fuse_w/2, fuse_y + 8, "Adaptive Gating\nFusion", ha='center', fontsize=12, fontweight='bold', color='#b8860b')
ax.text(fuse_x + fuse_w/2, fuse_y + 4, r"$H_{final} = \sum g_k \cdot H^{(k)}$", ha='center', fontsize=10, family='serif')

# 动态权重可视化 (条形图)
bar_x = fuse_x + 2
bar_w = 4
# 细粒度权重高 (假设当前是突发模式)
ax.add_patch(patches.Rectangle((bar_x, fuse_y - 8), bar_w, 4, facecolor=color_fine, edgecolor='black'))
ax.text(bar_x + bar_w/2, fuse_y - 10, "$g_{fine}$", ha='center', fontsize=9)
ax.text(bar_x + bar_w/2, fuse_y - 6, "High", ha='center', fontsize=8, color='white', fontweight='bold')

# 中粒度
ax.add_patch(patches.Rectangle((bar_x + 5, fuse_y - 8), bar_w, 3, facecolor=color_mid, edgecolor='black'))
ax.text(bar_x + 5 + bar_w/2, fuse_y - 10, "$g_{mid}$", ha='center', fontsize=9)

# 粗粒度权重低
ax.add_patch(patches.Rectangle((bar_x + 10, fuse_y - 8), bar_w, 1.5, facecolor=color_coarse, edgecolor='black'))
ax.text(bar_x + 10 + bar_w/2, fuse_y - 10, "$g_{coarse}$", ha='center', fontsize=9)
ax.text(bar_x + 10 + bar_w/2, fuse_y - 9, "Low", ha='center', fontsize=7, color='white')

ax.text(fuse_x + fuse_w/2, fuse_y - 14, "Dynamic Mode Switching", ha='center', fontsize=9, style='italic', color='gray')

# --- 5. 输出层 (Output) ---
out_x = 110  # 调整输出模块的位置
out_y = 50
out_box_x = out_x
out_box_y = 35
out_box_w = 10
out_box_h = 30

# 箭头
ax.annotate("", xy=(out_box_x, out_y), xytext=(fuse_x + fuse_w, fuse_y),
            arrowprops=dict(arrowstyle="->", color='black', lw=2))

# 输出框
ax.add_patch(patches.Rectangle((out_box_x, out_box_y), out_box_w, out_box_h, 
                                linewidth=2, edgecolor='black', facecolor='#e8f5e9'))

# 标题（放在框上方）
ax.text(out_box_x + out_box_w/2, out_box_y + out_box_h + 3, 
        "Prediction ($\hat{Y}_t$)", ha='center', fontsize=12, fontweight='bold')

# 绘制波形，确保完全在框内
margin = 0.5
t_out = np.linspace(0, 1, 60)
x_out = out_box_x + margin + t_out * (out_box_w - 2*margin)

# 创建带尖峰的波形
y_out_base = np.sin(t_out * 2 * np.pi) * 0.3
peak_pos = 0.6
y_out_peak = np.exp(-((t_out - peak_pos)**2) / 0.005) * 0.6
y_out = y_out_base + y_out_peak

# 映射到框内的 y 坐标
y_center = out_box_y + out_box_h/2
y_scale = (out_box_h - 6) / 2  # 留出上下各3单位的边距

# 真实值 (虚线，稍微偏移)
y_coords_truth = y_center + (y_out - 0.05) * y_scale
ax.plot(x_out, y_coords_truth, color='gray', linestyle='--', linewidth=1.5, label='Ground Truth')

# 预测值 (实线，高亮)
y_coords_pred = y_center + y_out * y_scale
ax.plot(x_out, y_coords_pred, color='green', linewidth=2.5, label='SDFormer Pred')

# 图例（放在框外右下方）
ax.legend(loc='upper left', bbox_to_anchor=(out_box_x - 0.5, out_box_y - 1), fontsize=9, frameon=True)

# 说明文字（放在框下方，不碰到框）
ax.text(out_box_x + out_box_w/2, out_box_y - 4, 
        "Peak Error Minimized!", ha='center', fontsize=10, color='green', fontweight='bold')

# --- 6. 底部图例说明 ---
legend_y = 5
ax.text(60, legend_y, "Key Innovations: 1. Multi-Granularity Independent Encoding  2. ATSR  3. Adaptive Gating Fusion (AGF)  4. Anomaly-Aware Normalization", 
        ha='center', fontsize=11, fontweight='bold', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round'))

plt.tight_layout()
plt.show()