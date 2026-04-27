import os
import time
import atexit

os.environ["PYGLET_HEADLESS"] = "True"

from multiprocessing.pool import Pool
from multiprocessing import TimeoutError, Process
from multiprocessing import get_context

import numpy as np
import trimesh
from scipy.spatial import cKDTree
import cadquery as cq


class NonDaemonProcess(Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NonDaemonPool(Pool):
    def Process(self, *args, **kwargs):
        proc = super(NonDaemonPool, self).Process(*args, **kwargs)
        proc.__class__ = NonDaemonProcess
        return proc


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def compute_iou(gt_mesh, pred_mesh):
    try:
        intersection_volume = 0
        for gt_mesh_i in gt_mesh.split():
            for pred_mesh_i in pred_mesh.split():
                intersection = gt_mesh_i.intersection(pred_mesh_i)
                volume = intersection.volume if intersection is not None else 0
                intersection_volume += volume

        gt_volume = sum(m.volume for m in gt_mesh.split())
        pred_volume = sum(m.volume for m in pred_mesh.split())
        union_volume = gt_volume + pred_volume - intersection_volume
        assert union_volume > 0
        return intersection_volume / union_volume
    except Exception:
        pass


def compute_cd(pred_mesh, gt_mesh, n_points=8192):
    gt_points, _ = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, _ = trimesh.sample.sample_surface(pred_mesh, n_points)
    gt_distance, _ = cKDTree(gt_points).query(pred_points, k=1)
    pred_distance, _ = cKDTree(pred_points).query(gt_points, k=1)
    cd = np.mean(np.square(gt_distance)) + np.mean(np.square(pred_distance))
    return cd


def transform_real_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    mesh.apply_scale(2.0 / max(mesh.extents))
    return mesh


def transform_gt_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_translation(-(mesh.bounds[0] + mesh.bounds[1]) / 2.0)
    extent = np.max(mesh.extents)
    if extent > 1e-7:
        mesh.apply_scale(1.0 / extent)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def transform_pred_mesh(mesh):
    if mesh is None:
        return None
    if mesh.bounds is None:
        return mesh
    mesh.apply_scale(1.0 / 200)
    mesh.apply_transform(trimesh.transformations.translation_matrix([0.5, 0.5, 0.5]))
    return mesh


def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)


def code_to_mesh_and_brep_less_safe(code_str):
    safe_ns = {"cq": cq}
    ns = safe_ns.copy()
    try:
        exec(code_str, ns)
        mesh = compound_to_mesh(ns["r"].val())
        return mesh
    except Exception as e:
        print(f"Error executing CadQuery code : {e}", flush=True)
        return None


def get_metrics_from_single_text(text, gt_file, n_points, normalize="fixed"):
    gt_file = os.path.abspath(gt_file)
    base_file = os.path.basename(gt_file).rsplit(".stl", 1)[0]

    try:
        pred_mesh = code_to_mesh_and_brep_less_safe(text)
    except Exception:
        return dict(file_name=base_file, cd=None, iou=None, auc=None)

    if pred_mesh is None:
        return dict(file_name=base_file, cd=None, iou=None, auc=None)

    cd, iou, auc = None, None, None
    try:
        gt_mesh = trimesh.load_mesh(gt_file)

        if normalize == "fixed":
            gt_mesh = transform_gt_mesh(gt_mesh)
            pred_mesh = transform_pred_mesh(pred_mesh)
        else:
            gt_mesh = transform_real_mesh(gt_mesh)
            pred_mesh = transform_real_mesh(pred_mesh)

        cd = compute_cd(gt_mesh, pred_mesh, n_points)
        iou = compute_iou(gt_mesh, pred_mesh)

    except Exception as e:
        print(f"error for {base_file}: {e}", flush=True)
        pass

    return dict(file_name=base_file, cd=cd, iou=iou, auc=auc)


POOL = None


def init_pool(max_workers=None):
    global POOL

    if max_workers is None:
        max_workers = max(1, min(os.cpu_count() or 1, 4))
    else:
        max_workers = max(1, int(max_workers))

    if POOL is None:
        print(f"Initializing POOL with {max_workers} workers", flush=True)
        ctx = get_context("spawn")
        POOL = NonDaemonPool(
            processes=max_workers,
            initializer=init_worker,
            context=ctx,
        )

    return POOL


def close_pool():
    global POOL
    if POOL is not None:
        try:
            POOL.close()
            POOL.join()
        finally:
            POOL = None


atexit.register(close_pool)


def timed_process_text(arg, timeout=60):
    """
    Supervisor function that workers will call.
    Spawns one child Process for real work, kills it on timeout.
    """
    ctx = get_context("fork")
    parent, child = ctx.Pipe(duplex=False)

    p = ctx.Process(target=_run_child, args=(child, arg))
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        parent.close()
        return "__TIMEOUT__"

    result = parent.recv() if parent.poll() else "__CRASH__"
    parent.close()
    return result


def _run_child(conn, arg):
    try:
        res = get_metrics_from_single_text(*arg)
        conn.send(res)
    finally:
        conn.close()


def get_metrics_from_texts(texts, meshes, max_workers=None, normalize="fixed"):
    if len(texts) == 0:
        return []

    pool = init_pool(max_workers=max_workers)
    print(f"[POOL] POOL size={pool._processes} pid={os.getpid()}", flush=True)

    n_points = 8192
    args = [
        (text, gt, n_points, normalize)
        for text, gt in zip(texts, meshes)
    ]

    async_results = [pool.apply_async(timed_process_text, args=(arg,)) for arg in args]

    results = []
    for res in async_results:
        output = res.get()
        if output == "__TIMEOUT__" or output == "__CRASH__":
            print(f"[{output}] metrics task computation ERROR, skipping", flush=True)
            results.append(dict(file_name=None, cd=None, iou=None, auc=None))
        else:
            results.append(output)

    return results


def compute_normals_metrics(gt_mesh, pred_mesh, tol=1, n_points=8192, visualize=False):
    """
    Input : normalized meshes
    computes the cosine similarity between the normals of the predicted mesh and the ground truth mesh.
    -> Done on a subset of points from the mesh point clouds
    Computes the area under the curve (AUC) of the angle distribution between the normals.
    Returns the aoc and mean_cos_sim
    """
    tol = pred_mesh.extents.max() * tol / 100

    gt_points, gt_face_indexes = trimesh.sample.sample_surface(gt_mesh, n_points)
    pred_points, pred_face_indexes = trimesh.sample.sample_surface(pred_mesh, n_points)

    gt_normals = gt_mesh.face_normals[gt_face_indexes]
    pred_normals = pred_mesh.face_normals[pred_face_indexes]

    tree = cKDTree(pred_points)
    neighbors = tree.query_ball_point(gt_points, r=tol)

    valid_pred_normals = []
    valid_gt_normals = []
    valid_gt_points = []
    valid_pred_points = []

    for i, idxs in enumerate(neighbors):
        if len(idxs) == 0:
            continue
        gn = gt_normals[i]
        pn_neighbors = pred_normals[idxs]

        valid_gt_normals.append(gn)
        dots = (pn_neighbors * gn).sum(axis=1)
        best_idx = np.argmax(dots)

        valid_pred_normals.append(pn_neighbors[best_idx])
        valid_gt_points.append(gt_points[i])
        valid_pred_points.append(pred_points[idxs[best_idx]])

    if len(valid_pred_normals) == 0:
        return 1.0, 0.0, 100

    valid_gt_normals = np.vstack(valid_gt_normals)
    valid_pred_normals = np.vstack(valid_pred_normals)
    valid_gt_points = np.vstack(valid_gt_points)
    valid_pred_points = np.vstack(valid_pred_points)

    nb_invalid = n_points - len(valid_pred_normals)
    per_invalid = nb_invalid / n_points * 100

    cos_sim = (valid_pred_normals * valid_gt_normals).sum(axis=1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    mean_cos_sim = np.mean(cos_sim)

    angles = np.arccos(cos_sim)
    angles = np.sort(angles)
    angles = np.concatenate((angles, np.full(nb_invalid, np.pi)))

    N = len(angles)
    cdf = np.arange(1, N + 1) / N

    from numpy import trapz
    x = np.concatenate(([0.0], angles, [np.pi]))
    y = np.concatenate(([0.0], cdf, [1.0]))
    auc_normalized = trapz(y, x) / np.pi

    return auc_normalized, mean_cos_sim, per_invalid