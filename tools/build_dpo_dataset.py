"""build_dpo_dataset.py — Build a preference-pair JSONL for DPO training.

For each STL in the input directory (or a hard-examples pkl), generates G
rollouts from the policy model, scores them with IoU reward, then writes one
JSONL record per example containing:
  y_w / y_l — a randomly sampled pair, ordered by reward
  ref_logp_w / ref_logp_l — log-probs under the reference (SFT) model

Skips examples where all G rollouts have identical reward (degenerate).
Supports resume: already-written gt_mesh_paths are skipped on restart.

Usage
-----
    # From hard-examples pkl (recommended — already filtered for informative examples)
    python3 tools/build_dpo_dataset.py \\
        --pkl ./data/mined/combined_hard.pkl \\
        --output ./data/dpo/combined_dpo.jsonl

    # From a raw STL directory
    python3 tools/build_dpo_dataset.py \\
        --data-dir ./data/cadrille_training/deepcad \\
        --output ./data/dpo/deepcad_dpo.jsonl \\
        --max-samples 10000

    # Use a separate reference model (default: same as policy)
    python3 tools/build_dpo_dataset.py \\
        --pkl ./data/mined/combined_hard.pkl \\
        --output ./data/dpo/combined_dpo.jsonl \\
        --ref-checkpoint ./checkpoints/cadrille-sft

Options
-------
    --checkpoint PATH       Policy model checkpoint (default: ./checkpoints/cadrille-sft)
    --ref-checkpoint PATH   Reference model checkpoint. If omitted, uses --checkpoint.
    --pkl PATH              Hard-examples pkl from rl/mine.py (preferred input)
    --data-dir PATH         Raw STL directory (alternative to --pkl)
    --output PATH           Output JSONL path (required)
    --modality {img,pc}     Input modality (default: img)
    --G N                   Rollouts per example (default: 8)
    --max-new-tokens N      Max tokens per rollout (default: 400)
    --temperature F         Sampling temperature (default: 0.8)
    --top-k N               Top-k sampling (default: 50)
    --reward-workers N      Parallel reward workers (default: 4)
    --max-samples N         Cap on number of STLs to process
    --batch-size N          Examples per generate() call (default: 4)
    --seed N                Random seed (default: 42)
    --resume                Skip already-written gt_mesh_paths
"""

import os
import sys
import json
import pickle
import random
import argparse
from glob import glob
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

from cadrille import Cadrille, collate
from rl.dataset import render_img
from rl.reward import compute_rewards_parallel


def parse_args():
    p = argparse.ArgumentParser(description='Build DPO preference-pair dataset')
    p.add_argument('--checkpoint',     default='./checkpoints/cadrille-sft')
    p.add_argument('--ref-checkpoint', default=None,
                   help='Reference model checkpoint. Defaults to --checkpoint.')
    p.add_argument('--pkl',            default=None, help='Hard-examples pkl from rl/mine.py')
    p.add_argument('--data-dir',       default=None, help='Raw STL directory')
    p.add_argument('--output',         required=True)
    p.add_argument('--modality',       default='img', choices=['img', 'pc'])
    p.add_argument('--G',              type=int,   default=8)
    p.add_argument('--max-new-tokens', type=int,   default=400)
    p.add_argument('--temperature',    type=float, default=0.8)
    p.add_argument('--top-k',          type=int,   default=50)
    p.add_argument('--reward-workers', type=int,   default=4)
    p.add_argument('--max-samples',    type=int,   default=None)
    p.add_argument('--batch-size',     type=int,   default=4)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument('--resume',         action='store_true')
    p.add_argument('--remap-prefix',   default=None,
                   help='Remap pkl paths: old_prefix:new_prefix (e.g., data/mined:data/deepcad_fusion_train)')
    p.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Base model or local path used to load processor/tokenizer",
    )
    return p.parse_args()


def load_model(checkpoint, device, base_model=None):
    """
    Load model weights from checkpoint, but always load the processor from the
    base Qwen2-VL model to match training/eval tokenization exactly.
    """
    print(f"Loading policy model from {checkpoint} ...")

    model = Cadrille.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        attn_implementation='flash_attention_2' if torch.cuda.is_available() else None,
    )
    model.eval()
    model.to(device)

    if base_model is None:
        raise ValueError(
            "build_dpo_dataset.py requires --base-model so processor loading matches training."
        )

    print(f"Loading processor from {base_model} ...")
    processor = AutoProcessor.from_pretrained(
        base_model,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side='left',
    )

    return model, processor


def prepare_item(stl_path, modality):
    """Load one STL → item dict, or None on failure."""
    file_name = os.path.splitext(os.path.basename(stl_path))[0]
    try:
        if modality == 'img':
            item = render_img(stl_path)
            item.update({'description': 'Generate cadquery code',
                         'file_name': file_name,
                         'gt_mesh_path': stl_path})
        else:
            import trimesh
            from dataset import mesh_to_point_cloud
            mesh = trimesh.load(stl_path)
            pc = mesh_to_point_cloud(mesh, 256)
            pc = (pc - 0.5) * 2
            item = {'point_cloud': pc,
                    'description': 'Generate cadquery code',
                    'file_name': file_name,
                    'gt_mesh_path': stl_path}
        return item
    except Exception as e:
        print(f'[prepare] {os.path.basename(stl_path)}: {e}')
        return None


@torch.no_grad()
def compute_ref_logprob(ref_model, processor, item, completion, device):
    """Compute mean sequence log-prob of completion under ref_model."""
    training_item = {k: v for k, v in item.items()
                     if k not in ('gt_mesh_path', 'file_name')}
    training_item['answer'] = completion
    batch = collate([training_item], processor=processor, n_points=256, eval=False)

    out = ref_model(
        input_ids=batch['input_ids'].to(device),
        attention_mask=batch['attention_mask'].to(device),
        labels=None,
        point_clouds=batch['point_clouds'].to(device),
        is_pc=batch['is_pc'].to(device),
        is_img=batch['is_img'].to(device),
        pixel_values_videos=(
            batch['pixel_values_videos'].to(device)
            if batch.get('pixel_values_videos') is not None else None),
        video_grid_thw=(
            batch['video_grid_thw'].to(device)
            if batch.get('video_grid_thw') is not None else None),
    )
    logp = Cadrille.compute_sequence_logprob(
        out.logits, batch['labels'].to(device), mean_reduction=True)
    return logp.squeeze(0).item()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # ── Collect STL paths ────────────────────────────────────────────────────
    if args.pkl:
        with open(args.pkl, 'rb') as f:
            records = pickle.load(f)
        stl_paths = [r['gt_mesh_path'] for r in records]
        if args.remap_prefix:
            old_prefix, new_prefix = args.remap_prefix.split(':', 1)
            remapped = []
            for p in stl_paths:
                if p.startswith(old_prefix):
                    basename = os.path.basename(p)
                    remapped.append(os.path.join(new_prefix, basename))
                else:
                    remapped.append(p)
            stl_paths = remapped
            print(f'Remapped paths: {old_prefix}/** → {new_prefix}/')
        print(f'Loaded {len(stl_paths)} examples from pkl: {args.pkl}')
    elif args.data_dir:
        stl_paths = sorted(glob(os.path.join(args.data_dir, '**', '*.stl'), recursive=True))
        print(f'Found {len(stl_paths)} STLs in {args.data_dir}')
    else:
        raise ValueError('Provide --pkl or --data-dir')

    if args.max_samples and len(stl_paths) > args.max_samples:
        rng = random.Random(args.seed)
        rng.shuffle(stl_paths)
        stl_paths = stl_paths[:args.max_samples]
        print(f'Capped to {len(stl_paths)} samples')

    # ── Resume: skip already-written paths ──────────────────────────────────
    done_paths = set()
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            for line in f:
                line = line.strip()
                if line:
                    done_paths.add(json.loads(line)['gt_mesh_path'])
        print(f'Resume: {len(done_paths)} already written, skipping.')
    stl_paths = [p for p in stl_paths if p not in done_paths]
    print(f'To process: {len(stl_paths)} examples')

    # ── Load models ──────────────────────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Loading policy model from {args.checkpoint} ...')
    policy_model, processor = load_model(
        args.checkpoint,
        device,
        args.base_model,
    )

    ref_ckpt = args.ref_checkpoint or args.checkpoint
    if args.ref_checkpoint is None:
        print('[warning] --ref-checkpoint not provided; using --checkpoint as the initial reference model.')
    if ref_ckpt == args.checkpoint:
        print('Reference model = policy model (same checkpoint)')
        ref_model = policy_model
    else:
        print(f'Loading reference model from {ref_ckpt} ...')
        ref_model, _ = load_model(ref_ckpt, device, args.base_model)

    eos_id = processor.tokenizer.eos_token_id
    pad_id = processor.tokenizer.pad_token_id or eos_id

    # ── Generation kwargs ────────────────────────────────────────────────────
    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=1.0,
        top_k=args.top_k,
        early_stopping=False,
    )
    blocked_token_ids = []
    for attr in ('video_token_id', 'image_token_id'):
        token_id = getattr(policy_model.config, attr, None)
        if token_id is not None:
            blocked_token_ids.append([token_id])
    if blocked_token_ids:
        gen_kwargs['bad_words_ids'] = blocked_token_ids

    # ── Main loop ────────────────────────────────────────────────────────────
    B = args.batch_size
    G = args.G
    n_written = 0
    n_degenerate = 0

    out_f = open(args.output, 'a')

    chunks = [stl_paths[i:i + B] for i in range(0, len(stl_paths), B)]
    pbar = tqdm(total=len(stl_paths), desc='build_dpo')

    for chunk in chunks:
        # Prepare items
        items_ok = []
        for p in chunk:
            item = prepare_item(p, args.modality)
            if item is not None:
                items_ok.append(item)

        if not items_ok:
            pbar.update(len(chunk))
            continue

        M = len(items_ok)

        # Collate prompt batch and generate G rollouts per example
        collate_items = []
        for it in items_ok:
            x = {k: v for k, v in it.items() if k != "gt_mesh_path"}
            x["file_name"] = it["file_name"]
            collate_items.append(x)

        batch = collate(collate_items, processor=processor, n_points=256, eval=True)
        prompt_len = batch['input_ids'].shape[1]

        # Expand batch M → M*G for batched generation
        expanded = {}
        for k in ('input_ids', 'attention_mask', 'point_clouds', 'is_pc', 'is_img',
                  'pixel_values_videos', 'video_grid_thw'):
            v = batch.get(k)
            if v is None:
                expanded[k] = None
            elif isinstance(v, torch.Tensor):
                expanded[k] = v.repeat_interleave(G, dim=0).to(device)
            else:
                expanded[k] = v

        # Disable GC for generate (same fix as cppo.py)
        had_gc = getattr(policy_model, 'is_gradient_checkpointing', False)
        if had_gc:
            policy_model.gradient_checkpointing_disable()
        try:
            if hasattr(policy_model, 'rope_deltas'):
                policy_model.rope_deltas = None
            with torch.no_grad():
                gen_ids = policy_model.generate(**expanded, **gen_kwargs)  # [M*G, full_len]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f'[OOM] chunk size {M}×G={G} — skipping chunk')
            pbar.update(len(chunk))
            continue
        finally:
            if had_gc:
                policy_model.gradient_checkpointing_enable()

        # Decode completions [M*G]
        completions = processor.batch_decode(
            gen_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)

        # Compute rewards
        gt_paths_flat = [it['gt_mesh_path'] for it in items_ok for _ in range(G)]
        rewards_flat = compute_rewards_parallel(
            completions, gt_paths_flat, workers=args.reward_workers)

        # Per-example: sample 2 candidates uniformly, then prefer the one
        # with the higher reward. This matches the paper's DPO pair construction.
        for i, item in enumerate(items_ok):
            rews = rewards_flat[i * G:(i + 1) * G]
            comps = completions[i * G:(i + 1) * G]

            rews_arr = np.array(rews, dtype=np.float32)
            if np.all(rews_arr == rews_arr[0]):
                n_degenerate += 1
                continue

            preferred_idx = None
            rejected_idx = None
            max_pair_tries = max(16, len(rews_arr) * 4)
            for _ in range(max_pair_tries):
                first, second = np.random.choice(len(rews_arr), size=2, replace=False)
                if rews_arr[first] == rews_arr[second]:
                    continue
                if rews_arr[first] > rews_arr[second]:
                    preferred_idx, rejected_idx = int(first), int(second)
                else:
                    preferred_idx, rejected_idx = int(second), int(first)
                break

            if preferred_idx is None or rejected_idx is None:
                n_degenerate += 1
                continue

            y_w = comps[preferred_idx]
            y_l = comps[rejected_idx]

            ref_logp_w = compute_ref_logprob(ref_model, processor, item, y_w, device)
            ref_logp_l = compute_ref_logprob(ref_model, processor, item, y_l, device)

            record = {
                'description':  item['description'],
                'file_name':    item['file_name'],
                'gt_mesh_path': item['gt_mesh_path'],
                'modality':     args.modality,
                'y_w':          y_w,
                'y_l':          y_l,
                'ref_logp_w':   round(ref_logp_w, 6),
                'ref_logp_l':   round(ref_logp_l, 6),
                'reward_w':     round(float(rews_arr[preferred_idx]),  4),
                'reward_l':     round(float(rews_arr[rejected_idx]), 4),
            }
            if args.modality == 'pc' and item.get('point_cloud') is not None:
                record['point_cloud'] = np.asarray(item['point_cloud'], dtype=np.float32).tolist()
            out_f.write(json.dumps(record) + '\n')
            out_f.flush()
            n_written += 1

        pbar.update(len(chunk))

    pbar.close()
    out_f.close()

    print(f'\nDone.')
    print(f'  Written:    {n_written} preference pairs → {args.output}')
    print(f'  Degenerate: {n_degenerate} examples skipped (all rewards equal)')


if __name__ == '__main__':
    main()
