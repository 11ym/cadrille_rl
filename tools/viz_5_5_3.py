"""Section 5.5.3 — Typical success and failure case analysis.

Uses RL checkpoint inference results from the 5.5.2 search JSON to identify:
  - Success cases: highest RL IoU
  - Failure cases: lowest RL IoU

Produces two separate figures:
  fig_success.png  — Input (4-view) | Ground Truth | Prediction
  fig_failure.png  — Input (4-view) | Ground Truth | Prediction

Usage:
    # Requires viz/vis_5_5_2/search.json from viz_5_5_2.py --search
    python tools/viz_5_5_3.py [--n 3] [--search-json viz/vis_5_5_2/search.json]
                               [--out viz/vis_5_5_3] [--cell 268] [--pad 6]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw

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


def generate_stl(code: str, out_stl: str) -> bool:
    full_code = code if 'exporters.export' in code else code + f"\ncq.exporters.export(r, '{out_stl}')"
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp = f.name
    try:
        subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=60)
        return os.path.exists(out_stl)
    finally:
        os.unlink(tmp)


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
    parser.add_argument('--n',           type=int, default=3,
                        help='Number of success/failure cases each')
    parser.add_argument('--search-json', default='viz/vis_5_5_2/search.json',
                        help='Path to search JSON from viz_5_5_2.py --search')
    parser.add_argument('--data-dir',    default='data/deepcad_test_mesh')
    parser.add_argument('--out',         default='viz/vis_5_5_3')
    parser.add_argument('--cell',        type=int, default=268)
    parser.add_argument('--pad',         type=int, default=6)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(base, args.out)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(base, args.search_json)) as f:
        data = json.load(f)

    # Sort by RL IoU
    rl_sorted = sorted(
        [(uid, d['rl']['iou'], d['rl']['code'], d['rl']['stl'])
         for uid, d in data['combined'].items() if d['rl']['iou'] is not None],
        key=lambda x: x[1], reverse=True
    )

    success = rl_sorted[:args.n]
    failure = rl_sorted[-args.n:]

    print('Success cases:')
    for uid, iou, _, _ in success:
        print(f'  {uid}  RL IoU={iou:.4f}')
    print('Failure cases:')
    for uid, iou, _, _ in failure:
        print(f'  {uid}  RL IoU={iou:.4f}')

    # Generate STLs and renders
    for uid, iou, code, gt_stl in success + failure:
        pred_stl = os.path.join(out, f'{uid}_rl_pred.stl')
        if not os.path.exists(pred_stl):
            ok = generate_stl(code, pred_stl)
            print(f'  STL {uid}: {"OK" if ok else "FAIL"}')

        gt_png   = os.path.join(out, f'{uid}_gt_single.png')
        pred_png = os.path.join(out, f'{uid}_rl_single.png')
        if not os.path.exists(gt_png):
            render_single_view(gt_stl, gt_png, args.cell)
        if not os.path.exists(pred_png):
            render_single_view(pred_stl, pred_png, args.cell)

    data_dir = os.path.join(base, args.data_dir)
    compose_grid(
        [[f'{data_dir}/{uid}_render.png',
          f'{out}/{uid}_gt_single.png',
          f'{out}/{uid}_rl_single.png']
         for uid, *_ in success],
        os.path.join(out, 'fig_success.png'), args.cell, args.pad)

    compose_grid(
        [[f'{data_dir}/{uid}_render.png',
          f'{out}/{uid}_gt_single.png',
          f'{out}/{uid}_rl_single.png']
         for uid, *_ in failure],
        os.path.join(out, 'fig_failure.png'), args.cell, args.pad)


if __name__ == '__main__':
    main()
