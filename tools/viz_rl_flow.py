"""Draw RL flow diagram (Figure 2.4 replacement).

环境 at center, 状态/动作/奖励/策略更新 at N/E/S/W.
Six arrows showing the RL loop. Legend on the right.

Usage:
    python tools/viz_rl_flow.py [--out viz/fig_rl_flow.png]
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch

FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_REG  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp_bold = fm.FontProperties(fname=FONT_BOLD)
fp_reg  = fm.FontProperties(fname=FONT_REG)


def draw_node(ax, cx, cy, w, h, label_zh, label_en, fc, ec):
    rect = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                          boxstyle='round,pad=0.12',
                          facecolor=fc, edgecolor=ec,
                          linewidth=2.5, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy + 0.10, label_zh, ha='center', va='center',
            fontproperties=fp_bold, fontsize=17, color='#1a1a1a', zorder=4)
    ax.text(cx, cy - 0.26, label_en, ha='center', va='center',
            fontproperties=fp_reg, fontsize=9, color='#999999', zorder=4)


def arrow(ax, x0, y0, x1, y1, color, lw=2.2, rad=0.0):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=lw,
                    mutation_scale=22,
                    connectionstyle=f'arc3,rad={rad}',
                ), zorder=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='viz/fig_rl_flow.png')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── Title ──────────────────────────────────────────────────────────────
    ax.text(5.2, 6.80, '图2.4  强化学习基本流程示意图',
            ha='center', va='top',
            fontproperties=fp_bold, fontsize=18, color='#111111')

    # ── Nodes ──────────────────────────────────────────────────────────────
    CX, CY = 5.2, 3.5          # center (环境)
    draw_node(ax, CX,      CY,       2.8, 1.4, '环境',   'Environment',  '#FDEAEA', '#D9534F')
    draw_node(ax, CX,      5.80,     2.2, 0.9, '状态',   'State',        '#EBF4FB', '#4A90D9')
    draw_node(ax, 1.60,    CY,       2.4, 0.9, '策略更新','Policy Update','#F3ECF9', '#9B59B6')
    draw_node(ax, 8.80,    CY,       2.2, 0.9, '动作',   'Action',       '#EBF9EE', '#4AAD62')
    draw_node(ax, CX,      1.20,     2.2, 0.9, '奖励',   'Reward',       '#FEF4E8', '#E08B2B')

    # ── Arrows ─────────────────────────────────────────────────────────────
    # 1. 环境 → 状态  (up, blue)
    arrow(ax, CX,           CY + 0.70,  CX,           5.80 - 0.45, '#4A90D9')
    # 2. 状态 → 策略更新  (diagonal down-left, blue)
    arrow(ax, CX - 1.10,    5.80 - 0.30, 1.60 + 1.20, CY + 0.38,  '#4A90D9')
    # 3. 策略更新 → 环境  (right, purple)
    arrow(ax, 1.60 + 1.20,  CY,          CX - 1.40,   CY,          '#9B59B6')
    # 4. 环境 → 动作  (right, green)
    arrow(ax, CX + 1.40,    CY,          8.80 - 1.10, CY,          '#4AAD62')
    # 5. 动作 → 奖励  (diagonal down-left, green)
    arrow(ax, 8.80 - 0.60,  CY - 0.45,  CX + 1.10,   1.20 + 0.35, '#4AAD62')
    # 6. 奖励 → 策略更新  (diagonal up-left, orange)
    arrow(ax, CX - 1.10,    1.20 + 0.30, 1.60 + 0.90, CY - 0.45,  '#E08B2B')

    # ── Legend ─────────────────────────────────────────────────────────────
    lx, ly = 10.6, 5.70
    lw, lh = 3.0, 4.0
    ax.add_patch(FancyBboxPatch((lx, ly - lh), lw, lh,
                                boxstyle='round,pad=0.15',
                                facecolor='#F7F7F7', edgecolor='#CCCCCC',
                                linewidth=1.5, zorder=3))
    ax.text(lx + lw / 2, ly - 0.40, '核心要素',
            ha='center', va='center',
            fontproperties=fp_bold, fontsize=14, color='#1a1a1a', zorder=4)

    items = [
        '状态：环境信息',
        '动作：当前决策',
        '奖励：动作优劣',
        '策略：决策规则',
        '更新：提高长期回报',
    ]
    for i, item in enumerate(items):
        ax.text(lx + 0.25, ly - 1.00 - i * 0.58, f'• {item}',
                ha='left', va='center',
                fontproperties=fp_reg, fontsize=11.5, color='#333333', zorder=4)

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
