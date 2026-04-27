"""Render STL mesh to a 2×2 grid PNG matching the dataset style.

Usage:
    python tools/render_stl.py input.stl output.png [--size 268] [--bg white]
"""
import argparse
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


GOLD = np.array([255, 255, 136]) / 255.0   # match training data color

VIEW_ANGLES = [
    (30, 45),
    (30, 225),
    (30, 135),
    (30, 315),
]


def shade_faces(mesh, elev_deg, azim_deg, ambient=0.35, diffuse=0.65):
    """Return per-face shading intensity based on a directional light."""
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    light = np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])
    normals = mesh.face_normals
    dot = np.clip(np.dot(normals, light), 0, 1)
    return ambient + diffuse * dot


def render_mesh_grid(stl_path, out_png, color=GOLD, bg='white', cell_px=268):
    mesh = trimesh.load(stl_path, force='mesh')
    if mesh.is_empty:
        raise ValueError(f'Empty mesh: {stl_path}')

    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(1.6 / max(mesh.extents))

    dpi = 100
    # Each cell gets its own figure, then we tile them
    cell_in = cell_px / dpi
    imgs = []
    for elev, azim in VIEW_ANGLES:
        fig = plt.figure(figsize=(cell_in, cell_in))
        fig.patch.set_facecolor(bg)
        ax = fig.add_axes([0, 0, 1, 1], projection='3d')

        verts = mesh.vertices[mesh.faces]
        intensities = shade_faces(mesh, elev, azim)
        face_colors = np.outer(intensities, color)
        face_colors = np.clip(face_colors, 0, 1)

        poly = Poly3DCollection(verts, linewidth=0)
        poly.set_facecolor(face_colors)
        ax.add_collection3d(poly)

        lim = 0.95
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)
        ax.axis('off')
        ax.set_facecolor(bg)
        # Remove margins
        ax.set_position([0, 0, 1, 1])

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        buf = buf[:, :, :3]
        imgs.append(buf)
        plt.close(fig)

    # Tile 2x2
    from PIL import Image
    cell_arr = [Image.fromarray(im).resize((cell_px, cell_px), Image.LANCZOS) for im in imgs]
    grid = Image.new('RGB', (cell_px * 2, cell_px * 2), color=(255, 255, 255))
    grid.paste(cell_arr[0], (0, 0))
    grid.paste(cell_arr[1], (cell_px, 0))
    grid.paste(cell_arr[2], (0, cell_px))
    grid.paste(cell_arr[3], (cell_px, cell_px))
    grid.save(out_png)
    plt.close('all')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stl', help='Input STL file')
    parser.add_argument('png', help='Output PNG file')
    parser.add_argument('--size', type=int, default=268, help='Cell size in pixels')
    parser.add_argument('--bg', default='white', help='Background color')
    args = parser.parse_args()
    render_mesh_grid(args.stl, args.png, cell_px=args.size, bg=args.bg)
    print(f'Saved: {args.png}')


if __name__ == '__main__':
    main()
