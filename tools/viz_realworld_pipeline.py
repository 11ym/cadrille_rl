"""Compose real-world CAD reconstruction pipeline figure.

Layout per row:  Input image → InstantMesh (mesh) → 4-view images → CAD output

Each row needs:
  --input   : real-world photo (any image)
  --mesh    : single-view render of InstantMesh mesh (.png)
  --fourview: 4-view composite render (.png, training format 268×268)
  --cad     : single-view render of cadrille CAD output (.png)

Multiple rows can be specified by repeating the group (see examples).

Usage:
    # Single row (water cup):
    python tools/viz_realworld_pipeline.py \\
        --rows "data/obj/水杯/杯子.jpg,viz/real_world/cup_mesh_single.png,data/obj/水杯/cup_render.png,viz/real_world/cup_cad_single.png" \\
        --out viz/fig_realworld.png

    # Two rows:
    python tools/viz_realworld_pipeline.py \\
        --rows \\
            "data/obj/水杯/杯子.jpg,viz/real_world/cup_mesh_single.png,data/obj/水杯/cup_render.png,viz/real_world/cup_cad_single.png" \\
            "data/obj/茶壶.jpg,viz/real_world/mesh_single.png,viz/real_world/mesh_correct_4view.png,viz/real_world/cad_single.png" \\
        --out viz/fig_realworld.png

    # Also supports --obj to generate mesh renders on the fly:
    python tools/viz_realworld_pipeline.py \\
        --obj data/obj/水杯/cup.stl \\
        --input-img data/obj/水杯/杯子.jpg \\
        --out viz/fig_realworld.png
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GOLD = np.array([255, 255, 136]) / 255.0


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_single_view(stl_path: str, out_png: str, cell_px: int = 300) -> bool:
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            return False
    except Exception:
        return False
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
    mesh.apply_scale(1.6 / max(mesh.extents))
    dpi = 100
    fig = plt.figure(figsize=(cell_px / dpi, cell_px / dpi))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    ax.set_facecolor('white')
    el, az = np.radians(25), np.radians(45)
    light = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    sh = 0.35 + 0.65 * np.clip(np.dot(mesh.face_normals, light), 0, 1)
    fc = np.clip(np.outer(sh, GOLD), 0, 1)
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    poly = Poly3DCollection(mesh.vertices[mesh.faces], linewidth=0)
    poly.set_facecolor(fc)
    ax.add_collection3d(poly)
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.95, 0.95); ax.set_zlim(-0.95, 0.95)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=45)
    ax.axis('off'); ax.set_position([0, 0, 1, 1])
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    Image.fromarray(buf).resize((cell_px, cell_px), Image.LANCZOS).save(out_png)
    return True


def render_4view_training_format(stl_path: str, out_png: str) -> bool:
    """Render 4 views with black background, 3px border, 268×268 composite."""
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import ImageOps
    try:
        mesh = trimesh.load(stl_path, force='mesh')
        if mesh.is_empty:
            return False
    except Exception:
        return False
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2)
    mesh.apply_scale(1.0 / max(mesh.extents))

    view_params = [(35, 45), (35, 225), (35, 135), (35, 315)]
    cell = 128

    imgs = []
    for elev, azim in view_params:
        dpi = 100
        fig = plt.figure(figsize=(cell / dpi, cell / dpi))
        fig.patch.set_facecolor('black')
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')
        ax.set_facecolor('black')
        el, az = np.radians(elev), np.radians(azim)
        light = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
        sh = 0.35 + 0.65 * np.clip(np.dot(mesh.face_normals, light), 0, 1)
        fc = np.clip(np.outer(sh, GOLD), 0, 1)
        poly = Poly3DCollection(mesh.vertices[mesh.faces], linewidth=0)
        poly.set_facecolor(fc)
        ax.add_collection3d(poly)
        lim = 0.55
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off'); ax.set_position([0, 0, 1, 1])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        plt.close(fig)
        imgs.append(Image.fromarray(buf).resize((cell, cell), Image.LANCZOS))

    bordered = [ImageOps.expand(im, border=3, fill='black') for im in imgs]
    combined = Image.fromarray(np.vstack([
        np.hstack([np.array(bordered[0]), np.array(bordered[1])]),
        np.hstack([np.array(bordered[2]), np.array(bordered[3])]),
    ]))
    combined.save(out_png)
    return True


# ---------------------------------------------------------------------------
# Image loading helpers
# ---------------------------------------------------------------------------

def load_square(path: str, size: int) -> Image.Image:
    img = Image.open(path).convert('RGB')
    w, h = img.size
    m = min(w, h)
    img = img.crop(((w - m) // 2, (h - m) // 2, (w + m) // 2, (h + m) // 2))
    return img.resize((size, size), Image.LANCZOS)


def load_4view_display(path: str, size: int) -> Image.Image:
    """Load 4-view training-format image, convert black bg to light gray."""
    raw = np.array(Image.open(path).convert('RGB'))
    arr = np.where(raw < 15, 240, raw).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# Figure composition
# ---------------------------------------------------------------------------

def compose(rows_data: list, out_path: str, cell: int = 300,
            pad: int = 24, arrow_w: int = 50):
    """
    rows_data: list of dicts with keys: input, mesh, fourview, cad
    """
    n_rows  = len(rows_data)
    n_cols  = 4
    hdr_h   = 72

    W = n_cols * cell + (n_cols - 1) * arrow_w + 2 * pad
    H = hdr_h + n_rows * cell + (n_rows - 1) * pad + 2 * pad

    fig, ax = plt.subplots(figsize=(W / 100, H / 100))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Column x-centers
    col_x = []
    x = pad
    for c in range(n_cols):
        col_x.append(x + cell / 2)
        x += cell + (arrow_w if c < n_cols - 1 else 0)

    # Headers
    headers = ['Input\nimage', 'InstantMesh\n(mesh)', '4-view\nimages', 'CAD\noutput']
    hdr_y = H - pad - hdr_h / 2
    for c, hdr in enumerate(headers):
        ax.text(col_x[c], hdr_y, hdr, ha='center', va='center',
                fontsize=15, fontweight='bold',
                fontfamily='DejaVu Sans', color='#111111')

    loaders = [load_square, load_square, load_4view_display, load_square]
    keys    = ['input', 'mesh', 'fourview', 'cad']

    for r, row in enumerate(rows_data):
        y_top = H - pad - hdr_h - r * (cell + pad)
        y_bot = y_top - cell
        cy    = (y_top + y_bot) / 2

        for c, (key, loader) in enumerate(zip(keys, loaders)):
            path = row.get(key, '')
            x0 = col_x[c] - cell / 2
            if path and os.path.exists(path):
                img = loader(path, cell)
                ax.imshow(np.array(img),
                          extent=[x0, x0 + cell, y_bot, y_top],
                          aspect='auto', origin='upper')
                ax.add_patch(plt.Rectangle((x0, y_bot), cell, cell,
                             fill=False, edgecolor='#CCCCCC', lw=0.8))
            else:
                ax.add_patch(plt.Rectangle((x0, y_bot), cell, cell,
                             facecolor='#EEEEEE', edgecolor='#CCCCCC'))
                ax.text(col_x[c], cy, 'missing', ha='center', va='center',
                        fontsize=9, color='red')

        # Arrows
        for c in range(n_cols - 1):
            x0a = col_x[c] + cell / 2 + 5
            x1a = col_x[c + 1] - cell / 2 - 5
            ax.annotate('', xy=(x1a, cy), xytext=(x0a, cy),
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=2.0, mutation_scale=20))

    fig.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'Saved: {out_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', nargs='+',
                        help='Each row as "input,mesh,fourview,cad" comma-separated paths')
    parser.add_argument('--obj',       help='STL/OBJ path to auto-render (single row mode)')
    parser.add_argument('--input-img', help='Real-world photo for --obj mode')
    parser.add_argument('--cad-stl',   help='CAD output STL for --obj mode')
    parser.add_argument('--out', default='viz/fig_realworld.png')
    parser.add_argument('--cell', type=int, default=300)
    args = parser.parse_args()

    base    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out     = os.path.join(base, args.out)
    tmpdir  = os.path.join(base, 'viz', 'real_world')
    os.makedirs(tmpdir, exist_ok=True)

    rows_data = []

    if args.obj:
        # Auto-render mode
        stl = args.obj if args.obj.endswith('.stl') else args.obj
        mesh_png  = os.path.join(tmpdir, 'auto_mesh_single.png')
        four_png  = os.path.join(tmpdir, 'auto_4view.png')
        render_single_view(stl, mesh_png, args.cell)
        render_4view_training_format(stl, four_png)
        cad_png = ''
        if args.cad_stl and os.path.exists(args.cad_stl):
            cad_png = os.path.join(tmpdir, 'auto_cad_single.png')
            render_single_view(args.cad_stl, cad_png, args.cell)
        rows_data.append({
            'input':    args.input_img or '',
            'mesh':     mesh_png,
            'fourview': four_png,
            'cad':      cad_png,
        })

    elif args.rows:
        for row_str in args.rows:
            parts = [p.strip() for p in row_str.split(',')]
            while len(parts) < 4:
                parts.append('')
            rows_data.append({
                'input':    os.path.join(base, parts[0]) if parts[0] else '',
                'mesh':     os.path.join(base, parts[1]) if parts[1] else '',
                'fourview': os.path.join(base, parts[2]) if parts[2] else '',
                'cad':      os.path.join(base, parts[3]) if parts[3] else '',
            })

    else:
        # Default: water cup example
        rows_data = [{
            'input':    os.path.join(base, 'data/obj/水杯/杯子.jpg'),
            'mesh':     os.path.join(base, 'viz/real_world/cup_mesh_single.png'),
            'fourview': os.path.join(base, 'data/obj/水杯/cup_render.png'),
            'cad':      os.path.join(base, 'viz/real_world/cup_cad_single.png'),
        }]

    compose(rows_data, out, cell=args.cell)


if __name__ == '__main__':
    main()
