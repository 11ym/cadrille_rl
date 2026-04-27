"""Section 5.5.1 — Multi-modality input visualization.

For each modality (image, point cloud, text), renders a grid figure with columns:
  Input | Ground Truth | Prediction

Requires pre-existing inference results in viz/vis_5_5_1/{group}/.
Run tools/infer_cases.py first to generate prediction STLs, then this script.

Usage:
    python tools/viz_5_5_1.py [--cell 268] [--pad 6] [--out viz/vis_5_5_1]
"""
import argparse
import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLD = np.array([255, 255, 136]) / 255.0
ELEV, AZIM = 25, 45


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _shade(mesh, elev, azim, ambient=0.35, diffuse=0.65):
    el, az = np.radians(elev), np.radians(azim)
    light = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    return ambient + diffuse * np.clip(np.dot(mesh.face_normals, light), 0, 1)


def render_single_view(stl_path: str, out_png: str, cell_px: int = 268) -> bool:
    import trimesh
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            return False
    except Exception:
        return False
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(1.6 / max(mesh.extents))

    dpi = 100
    fig = plt.figure(figsize=(cell_px / dpi, cell_px / dpi))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    face_colors = np.clip(np.outer(_shade(mesh, ELEV, AZIM), GOLD), 0, 1)
    poly = Poly3DCollection(mesh.vertices[mesh.faces], linewidth=0)
    poly.set_facecolor(face_colors)
    ax.add_collection3d(poly)
    lim = 0.95
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.axis('off'); ax.set_facecolor('white'); ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    Image.fromarray(buf).resize((cell_px, cell_px), Image.LANCZOS).save(out_png)
    return True


def render_pc_view(point_cloud: np.ndarray, out_png: str, cell_px: int = 268):
    dpi = 100
    fig = plt.figure(figsize=(cell_px / dpi, cell_px / dpi))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2],
               s=1.5, c='steelblue', alpha=0.7)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)
    ax.axis('off'); ax.set_facecolor('white')
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    Image.fromarray(buf).resize((cell_px, cell_px), Image.LANCZOS).save(out_png)


def make_text_card(text: str, out_png: str, cell_px: int = 268):
    img = Image.new('RGB', (cell_px, cell_px), (248, 248, 252))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    except Exception:
        font = ImageFont.load_default()
    words = text.split()
    lines, line = [], ''
    for w in words:
        test = (line + ' ' + w).strip()
        if draw.textbbox((0, 0), test, font=font)[2] > cell_px - 16:
            if line:
                lines.append(line)
            line = w
        else:
            line = test
    if line:
        lines.append(line)
    y = 10
    for ln in lines:
        if y + 14 > cell_px - 8:
            draw.text((8, y), '...', fill=(100, 100, 100), font=font)
            break
        draw.text((8, y), ln, fill=(30, 30, 30), font=font)
        y += 14
    draw.rectangle([0, 0, cell_px - 1, cell_px - 1], outline=(180, 180, 200), width=1)
    img.save(out_png)


# ---------------------------------------------------------------------------
# Grid composition
# ---------------------------------------------------------------------------

def compose_grid(rows, out_path: str, cell_px: int = 268, pad: int = 6):
    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    W = n_cols * cell_px + (n_cols + 1) * pad
    H = n_rows * cell_px + (n_rows + 1) * pad
    canvas = Image.new('RGB', (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(rows):
        for c, path in enumerate(row):
            x = pad + c * (cell_px + pad)
            y = pad + r * (cell_px + pad)
            if path and os.path.exists(path):
                img = Image.open(path).convert('RGB').resize((cell_px, cell_px), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle([x, y, x + cell_px, y + cell_px], fill=(210, 210, 210))
    canvas.save(out_path, dpi=(150, 150))
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell', type=int, default=268, help='Cell size in pixels')
    parser.add_argument('--pad',  type=int, default=6,   help='Padding between cells')
    parser.add_argument('--out',  default='viz/vis_5_5_1', help='Output directory')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(base, args.out)
    cell, pad = args.cell, args.pad

    # ── DeepCAD image ────────────────────────────────────────────────────────
    compose_grid([
        [f'{base}/data/deepcad_test_mesh/00001926_render.png',
         f'{out}/deepcad_img/00001926_gt_single.png',
         f'{out}/deepcad_img/00001926_pred_single.png'],
        [f'{base}/data/deepcad_test_mesh/00002718_render.png',
         f'{out}/deepcad_img/00002718_gt_single.png',
         f'{out}/deepcad_img/00002718_pred_single.png'],
    ], f'{out}/fig_deepcad_img.png', cell, pad)

    # ── Fusion360 image ───────────────────────────────────────────────────────
    compose_grid([
        [f'{base}/data/fusion360_test_mesh/128996_42176b10_0000_render.png',
         f'{out}/fusion360_img/128996_42176b10_0000_gt_single.png',
         f'{out}/fusion360_img/128996_42176b10_0000_pred_single.png'],
        [f'{base}/data/fusion360_test_mesh/34769_44655d03_0006_render.png',
         f'{out}/fusion360_img/34769_44655d03_0006_gt_single.png',
         f'{out}/fusion360_img/34769_44655d03_0006_pred_single.png'],
    ], f'{out}/fig_fusion360_img.png', cell, pad)

    # ── DeepCAD point cloud ───────────────────────────────────────────────────
    compose_grid([
        [f'{out}/deepcad_pc/00000093_pc_input.png',
         f'{out}/deepcad_pc/00000093_gt_single.png',
         f'{out}/deepcad_pc/00000093_pred_single.png'],
        [f'{out}/deepcad_pc/00003166_pc_input.png',
         f'{out}/deepcad_pc/00003166_gt_single.png',
         f'{out}/deepcad_pc/00003166_pred_single.png'],
    ], f'{out}/fig_deepcad_pc.png', cell, pad)

    # ── Fusion360 point cloud ─────────────────────────────────────────────────
    compose_grid([
        [f'{out}/fusion360_pc/128996_42176b10_0000_pc_input.png',
         f'{out}/fusion360_pc/128996_42176b10_0000_gt_single.png',
         f'{out}/fusion360_pc/128996_42176b10_0000_pred_single.png'],
        [f'{out}/fusion360_pc/34769_44655d03_0006_pc_input.png',
         f'{out}/fusion360_pc/34769_44655d03_0006_gt_single.png',
         f'{out}/fusion360_pc/34769_44655d03_0006_pred_single.png'],
    ], f'{out}/fig_fusion360_pc.png', cell, pad)

    # ── Text2CAD ──────────────────────────────────────────────────────────────
    compose_grid([
        [f'{out}/text2cad/00659752_text_input.png',
         f'{out}/text2cad/00659752_gt_single.png',
         f'{out}/text2cad/00659752_pred_single.png'],
        [f'{out}/text2cad/00857066_text_input.png',
         f'{out}/text2cad/00857066_gt_single.png',
         f'{out}/text2cad/00857066_pred_single.png'],
    ], f'{out}/fig_text2cad.png', cell, pad)


if __name__ == '__main__':
    main()
