"""Section 5.5.2 — SFT vs RL prediction comparison.

Samples DeepCAD test meshes, runs both SFT and RL inference, finds cases
where RL most improves over SFT, and composes a comparison figure:
  Input (4-view) | Ground Truth | SFT Prediction | RL Prediction

Usage:
    # Step 1: search for hard cases (runs both models, saves search JSON)
    python tools/viz_5_5_2.py --search --n-sample 40 --out viz/vis_5_5_2

    # Step 2: compose figure from existing search results
    python tools/viz_5_5_2.py --compose --top 3 --out viz/vis_5_5_2
"""
import argparse
import json
import os
import random
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
# Inference helpers
# ---------------------------------------------------------------------------

def score_code(code, gt_path):
    from rl.reward import _get_worker_path
    payload = json.dumps({'code_str': code, 'gt_mesh_path': gt_path, 'compute_chamfer': False})
    try:
        proc = subprocess.run([sys.executable, _get_worker_path()],
                              input=payload, capture_output=True, text=True, timeout=30)
        return json.loads(proc.stdout.strip()) if proc.stdout.strip() else {'iou': None}
    except Exception:
        return {'iou': None}


def infer_batch(model, processor, stl_paths, device):
    import torch
    from rl.dataset import render_img
    from cadrille import collate
    results = {}
    for stl in stl_paths:
        uid = os.path.splitext(os.path.basename(stl))[0]
        item = {'description': 'Generate cadquery code', 'file_name': uid, 'gt_mesh_path': stl}
        item.update(render_img(stl))
        batch = collate([item], processor=processor, n_points=256, eval=True)
        prompt_len = batch['input_ids'].shape[1]
        if hasattr(model, 'rope_deltas'):
            model.rope_deltas = None
        gen_input = {k: batch[k].to(device)
                     for k in ['input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img']}
        if batch.get('pixel_values_videos') is not None:
            gen_input['pixel_values_videos'] = batch['pixel_values_videos'].to(device)
        if batch.get('video_grid_thw') is not None:
            gen_input['video_grid_thw'] = batch['video_grid_thw'].to(device)
        with torch.no_grad():
            ids = model.generate(**gen_input, max_new_tokens=1024, do_sample=False)
        code = processor.decode(ids[0, prompt_len:], skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        scored = score_code(code, stl)
        results[uid] = {'code': code, 'iou': scored.get('iou'), 'stl': stl}
        print(f'  {uid}: IoU={results[uid]["iou"]}')
    return results


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--search',   action='store_true', help='Run model inference search')
    parser.add_argument('--compose',  action='store_true', help='Compose figure from existing search JSON')
    parser.add_argument('--n-sample', type=int, default=40, help='Cases to sample for search')
    parser.add_argument('--top',      type=int, default=3,  help='Top N cases to include in figure')
    parser.add_argument('--sft-ckpt', default='checkpoints/cadrille_sft')
    parser.add_argument('--rl-ckpt',  default='checkpoints/cadrille-rl')
    parser.add_argument('--data-dir', default='data/deepcad_test_mesh')
    parser.add_argument('--out',      default='viz/vis_5_5_2')
    parser.add_argument('--cell',     type=int, default=268)
    parser.add_argument('--pad',      type=int, default=6)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out  = os.path.join(base, args.out)
    os.makedirs(out, exist_ok=True)
    search_json = os.path.join(out, 'search.json')

    if args.search:
        import torch
        from transformers import AutoProcessor
        from cadrille import Cadrille

        from glob import glob
        all_stls = sorted(glob(os.path.join(base, args.data_dir, '*.stl')))
        random.seed(99)
        random.shuffle(all_stls)
        candidates = all_stls[:args.n_sample]

        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

        print('=== SFT ===')
        proc_sft = AutoProcessor.from_pretrained(os.path.join(base, args.sft_ckpt),
            min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
        mdl_sft = Cadrille.from_pretrained(os.path.join(base, args.sft_ckpt),
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
        mdl_sft.eval()
        device = next(mdl_sft.parameters()).device
        sft_res = infer_batch(mdl_sft, proc_sft, candidates, device)
        del mdl_sft; torch.cuda.empty_cache()

        print('=== RL ===')
        proc_rl = AutoProcessor.from_pretrained(os.path.join(base, args.rl_ckpt),
            min_pixels=256*28*28, max_pixels=1280*28*28, padding_side='left')
        mdl_rl = Cadrille.from_pretrained(os.path.join(base, args.rl_ckpt),
            torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2', device_map='auto')
        mdl_rl.eval()
        rl_res = infer_batch(mdl_rl, proc_rl, candidates, device)
        del mdl_rl; torch.cuda.empty_cache()

        rows = []
        for uid in sft_res:
            s, r = sft_res[uid]['iou'], rl_res[uid]['iou']
            if s is not None and r is not None:
                rows.append({'uid': uid, 'sft': s, 'rl': r, 'gain': r - s,
                             'stl': sft_res[uid]['stl']})
        rows.sort(key=lambda x: x['gain'], reverse=True)
        combined = {uid: {'sft': sft_res[uid], 'rl': rl_res[uid]} for uid in sft_res}
        with open(search_json, 'w') as f:
            json.dump({'rows': rows, 'combined': combined}, f, indent=2)
        print(f'Saved: {search_json}')
        for row in rows[:10]:
            print(f"  {row['uid']}  SFT={row['sft']:.4f}  RL={row['rl']:.4f}  gain={row['gain']:+.4f}")

    if args.compose:
        with open(search_json) as f:
            data = json.load(f)

        picks = [r['uid'] for r in data['rows'][:args.top]]
        print(f'Composing figure for: {picks}')

        for uid in picks:
            for model_name in ['sft', 'rl']:
                code = data['combined'][uid][model_name]['code']
                stl_out = os.path.join(out, f'{uid}_{model_name}_pred.stl')
                if not os.path.exists(stl_out):
                    ok = generate_stl(code, stl_out)
                    print(f'  STL {uid} {model_name}: {"OK" if ok else "FAIL"}')
                gt_stl = data['combined'][uid]['rl']['stl']
                for src, dst in [(gt_stl, f'{out}/{uid}_gt_single.png'),
                                 (f'{out}/{uid}_sft_pred.stl', f'{out}/{uid}_sft_single.png'),
                                 (f'{out}/{uid}_rl_pred.stl',  f'{out}/{uid}_rl_single.png')]:
                    if not os.path.exists(dst):
                        render_single_view(src, dst, args.cell)

        compose_grid(
            [[f'{base}/data/deepcad_test_mesh/{uid}_render.png',
              f'{out}/{uid}_gt_single.png',
              f'{out}/{uid}_sft_single.png',
              f'{out}/{uid}_rl_single.png']
             for uid in picks],
            os.path.join(out, 'fig_sft_vs_rl.png'), args.cell, args.pad)


if __name__ == '__main__':
    main()
