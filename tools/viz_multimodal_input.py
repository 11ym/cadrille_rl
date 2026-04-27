"""Draw multimodal input figure using actual dataset examples.

Data-driven style: 点云 | 图像 | 文本, three panels, no card backgrounds.

Usage:
    python tools/viz_multimodal_input.py [--uid 00614285] [--out viz/fig_multimodal_input.png]
"""
import argparse
import os
import pickle
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image

FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_REG  = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp_bold = fm.FontProperties(fname=FONT_BOLD)
fp_reg  = fm.FontProperties(fname=FONT_REG)

PC_COLOR = np.array([0x5B, 0x7F, 0xBE]) / 255.0


def sample_and_project(stl_path, n=6000):
    import trimesh
    mesh = trimesh.load(stl_path, force='mesh')
    pts, _ = trimesh.sample.sample_surface(mesh, n)
    center = (pts.max(0) + pts.min(0)) / 2
    pts -= center
    s = pts.max()
    if s > 0:
        pts /= s
    el, az = np.radians(30), np.radians(45)
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az),  np.cos(az), 0],
                   [0, 0, 1]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    proj = (Rx @ Rz @ pts.T).T
    return proj[:, 0], proj[:, 2], proj[:, 1]


def draw_pc(ax, stl_path):
    x2d, y2d, depth = sample_and_project(stl_path)
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    alphas = 0.25 + 0.75 * d_norm
    sizes  = 1.5 + 5.0 * d_norm
    order  = np.argsort(depth)

    rgba = np.ones((len(order), 4))
    rgba[:, :3] = PC_COLOR
    rgba[:, 3]  = alphas[order]

    ax.scatter(x2d[order], y2d[order], c=rgba, s=sizes[order], linewidths=0)
    # Tight limits with small margin
    margin = 0.08
    ax.set_xlim(x2d.min() - margin, x2d.max() + margin)
    ax.set_ylim(y2d.min() - margin, y2d.max() + margin)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor('white')


def recolor_to_blue(img_np):
    """Rotate yellow hue (~60°) to steel blue (~210°) in the image."""
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    img_f = img_np.astype(np.float32) / 255.0
    hsv = rgb_to_hsv(img_f)
    mask = hsv[:, :, 1] > 0.08          # only saturated (non-white) pixels
    hsv[:, :, 0] = np.where(mask, (hsv[:, :, 0] + 0.41) % 1.0, hsv[:, :, 0])
    return (hsv_to_rgb(hsv) * 255).astype(np.uint8)


def draw_img_grid(ax, render_png, pad=4):
    """Split the 4-view composite into 2×2, recolor to blue, light-gray borders."""
    img = recolor_to_blue(np.array(Image.open(render_png).convert('RGB')))
    h, w = img.shape[:2]
    mh, mw = h // 2, w // 2
    quads = [img[:mh, :mw], img[:mh, mw:], img[mh:, :mw], img[mh:, mw:]]
    cell   = mh
    border = 2
    border_color = np.array([190, 190, 190], dtype=np.uint8)
    canvas_size = 2 * cell + 3 * pad + 4 * border
    canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 255

    for (row, col), q in zip([(0,0),(0,1),(1,0),(1,1)], quads):
        y0 = pad + row * (cell + pad + 2 * border) + border
        x0 = pad + col * (cell + pad + 2 * border) + border
        canvas[y0 - border:y0 + cell + border,
               x0 - border:x0 + cell + border] = border_color
        canvas[y0:y0 + cell, x0:x0 + cell] = q[:cell, :cell]

    ax.imshow(canvas)
    ax.axis('off')
    ax.set_facecolor('white')


def draw_text(ax, description, max_chars=38, max_lines=6):
    lines = textwrap.wrap(description, width=max_chars)
    shown = lines[:max_lines]
    if len(lines) > max_lines:
        shown.append('...')
    text = '\n'.join(shown)
    ax.text(0.05, 0.55, text,
            ha='left', va='center',
            fontproperties=fp_reg, fontsize=13,
            color='#333333', linespacing=1.80,
            style='italic',
            transform=ax.transAxes)
    ax.axis('off')
    ax.set_facecolor('white')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--uid',      default='00308247')
    parser.add_argument('--data-dir', default='data/deepcad_test_mesh')
    parser.add_argument('--pkl',      default='data/text2cad/test.pkl')
    parser.add_argument('--out',      default='viz/fig_multimodal_input.png')
    args = parser.parse_args()

    base     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base, args.data_dir)
    out_path = os.path.join(base, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    stl_path    = os.path.join(data_dir, f'{args.uid}.stl')
    render_path = os.path.join(data_dir, f'{args.uid}_render.png')

    # Load text description
    description = '(no description)'
    pkl_path = os.path.join(base, args.pkl)
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            items = pickle.load(f)
        for item in items:
            if item.get('uid') == args.uid:
                description = item.get('description', description)
                break

    print(f'UID: {args.uid}')
    print(f'Description: {description[:120]}...')

    # ── Figure layout ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5.0))
    fig.patch.set_facecolor('white')

    panel_defs = [
        ('点云',   0.02, 0.30),
        ('图像',   0.36, 0.26),
        ('文本',   0.65, 0.34),
    ]
    height = 0.72
    bottom = 0.05

    for label, lx, lw in panel_defs:
        fig.text(lx + lw / 2, 0.93, label,
                 ha='center', va='top',
                 fontproperties=fp_bold, fontsize=22, color='#1a1a1a')

    # PC
    ax_pc = fig.add_axes([panel_defs[0][1], bottom, panel_defs[0][2], height])
    draw_pc(ax_pc, stl_path)

    # Images
    ax_img = fig.add_axes([panel_defs[1][1], bottom, panel_defs[1][2], height])
    draw_img_grid(ax_img, render_path)

    # Text
    ax_txt = fig.add_axes([panel_defs[2][1], bottom, panel_defs[2][2], height])
    draw_text(ax_txt, description)

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
