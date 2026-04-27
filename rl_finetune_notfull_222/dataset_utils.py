import os
import pickle
import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset

from utils_cadrille import transform_real_mesh
import trimesh
from PIL import Image
from pytorch3d.ops import sample_farthest_points


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))


def mesh_to_point_cloud(mesh, n_points=256, n_pre_points=8192):
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    return np.asarray(vertices[ids])


def _join_project_relative(path_str: str) -> str:
    path_str = path_str.lstrip("./")
    return os.path.join(PROJECT_ROOT, path_str)


def resolve_mesh_path(root: str, item: dict) -> str:
    """
    兼容多种 pkl 格式：
    1) {'mesh_path': '00000172.stl'}
    2) {'gt_mesh_path': './data/deepcad_train_mesh/00000172.stl'}
    3) {'file_name': '00000172', 'dataset': 'deepcad_train_mesh'}
    """
    if "mesh_path" in item:
        mesh_name = item["mesh_path"]
        if os.path.isabs(mesh_name):
            return mesh_name
        if mesh_name.startswith("./") or mesh_name.startswith("data/"):
            return _join_project_relative(mesh_name)
        return os.path.join(root, mesh_name)

    if "gt_mesh_path" in item:
        mesh_name = item["gt_mesh_path"]
        if os.path.isabs(mesh_name):
            return mesh_name
        return _join_project_relative(mesh_name)

    if "file_name" in item:
        mesh_name = item["file_name"]
        if not mesh_name.endswith(".stl"):
            mesh_name = mesh_name + ".stl"

        if os.path.isabs(mesh_name):
            return mesh_name

        if "dataset" in item and item["dataset"]:
            return os.path.join(PROJECT_ROOT, "data", item["dataset"], mesh_name)

        return os.path.join(root, mesh_name)

    if "uid" in item:
        mesh_name = item["uid"] + ".stl"
        return os.path.join(root, mesh_name)

    raise KeyError(f"Unsupported annotation keys: {list(item.keys())}")


def resolve_description(item: dict) -> str:
    return item.get("description", "Generate cadquery code")


def resolve_images(mesh_path: str, item: dict | None = None):
    """
    优先使用 pkl 自带的 video；
    否则回退到同名 *_render.png
    """
    if item is not None and "video" in item and item["video"] is not None:
        video = item["video"]
        if isinstance(video, list):
            return video
        return [video]

    if item is not None and "image" in item and item["image"] is not None:
        image = item["image"]
        if isinstance(image, list):
            return image
        return [image]

    img_path = os.path.splitext(mesh_path)[0] + "_render.png"
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"rendered image not found: {img_path}")

    with Image.open(img_path) as im:
        image = im.convert("RGB").copy()
    return [image]


class TrainDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, item["mesh_path"]))
        with open(os.path.join(self.path, item["py_path"]), "r") as f:
            py_string = f.read()

        item = self.tokenizer("<|im_start|>" + py_string + "<|endoftext|>")
        item["input_ids"] = [self.tokenizer.pad_token_id] * self.n_points + item["input_ids"]
        item["attention_mask"] = [-1] * self.n_points + item["attention_mask"]

        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item["point_cloud"] = point_cloud.astype(np.float32)
        item["mesh"] = mesh
        return item


class TrainRLDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, item["mesh_path"]))
        with open(os.path.join(self.path, item["py_path"]), "r") as f:
            py_string = f.read()

        item = self.tokenizer("<|im_start|>")
        item["input_ids"] = [self.tokenizer.pad_token_id] * self.n_points + item["input_ids"]
        item["attention_mask"] = [-1] * self.n_points + item["attention_mask"]
        item["py_string"] = py_string

        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item["point_cloud"] = point_cloud.astype(np.float32)
        item["mesh"] = mesh
        return item


class TrainDPODataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points, normalize_std):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        self.normalize_std = normalize_std
        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        orig_item = self.annotations[index]
        mesh = trimesh.load(os.path.join(self.path, orig_item["mesh_path"]))
        with open(os.path.join(self.path, orig_item["py_path"]), "r") as f:
            py_string = f.read()
        with open(os.path.join(self.path, orig_item["ious"]), "rb") as f:
            ious = np.load(f)

        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w

        item = self.tokenizer("<|im_start|>")
        item["input_ids"] = [self.tokenizer.pad_token_id] * self.n_points + item["input_ids"]
        item["attention_mask"] = [-1] * self.n_points + item["attention_mask"]
        item["py_string"] = py_string

        with open(os.path.join(self.path, orig_item[w]), "r") as f:
            item["py_string_w"] = f.read() + "<|endoftext|>"
        with open(os.path.join(self.path, orig_item[l]), "r") as f:
            item["py_string_l"] = f.read() + "<|endoftext|>"

        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud[:, :3] = point_cloud[:, :3] / self.normalize_std
        item["point_cloud"] = point_cloud.astype(np.float32)
        item["ious"] = ious
        item["mesh"] = mesh
        return item


class FilteredDataset(Dataset):
    def __init__(self, dataset, filtered_idxs):
        super().__init__()
        self.dataset = dataset
        self.filtered_idxs = filtered_idxs

    def __len__(self):
        return len(self.filtered_idxs)

    def __getitem__(self, index):
        return self.dataset[self.filtered_idxs[index]]


class RealDataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points=256):
        super().__init__()
        self.start_token_id = tokenizer("<|im_start|>")["input_ids"][0]
        self.pad_token_id = tokenizer.pad_token_id
        self.n_points = n_points
        self.path = path

        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        try:
            item = self.annotations[idx]
            mesh_path = resolve_mesh_path(self.path, item)
            mesh = trimesh.load_mesh(mesh_path)
            mesh = transform_real_mesh(mesh)

            vertices = mesh_to_point_cloud(mesh, self.n_points)
            point_cloud = vertices
            input_ids = [self.pad_token_id] * self.n_points + [self.start_token_id]
            attention_mask = [-1] * self.n_points + [1]
            return {
                "point_cloud": point_cloud.astype(np.float32),
                "input_ids": np.array(input_ids),
                "attention_mask": np.array(attention_mask),
                "mesh_path": mesh_path,
                "mesh": mesh,
            }
        except Exception:
            return self[(idx + 1) % len(self)]


class RealDPODataset(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points):
        super().__init__()
        self.path = path
        self.tokenizer = tokenizer
        self.n_points = n_points
        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        orig_item = self.annotations[index]
        mesh_path = resolve_mesh_path(self.path, orig_item)
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        with open(os.path.join(self.path, orig_item["ious"]), "rb") as f:
            ious = np.load(f)

        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w

        item = self.tokenizer("<|im_start|>")
        item["input_ids"] = [self.tokenizer.pad_token_id] * self.n_points + item["input_ids"]
        item["attention_mask"] = [-1] * self.n_points + item["attention_mask"]
        item["mesh"] = mesh

        with open(os.path.join(self.path, orig_item[w]), "r") as f:
            item["py_string_w"] = f.read() + "<|endoftext|>"
        with open(os.path.join(self.path, orig_item[l]), "r") as f:
            item["py_string_l"] = f.read() + "<|endoftext|>"

        vertices = mesh_to_point_cloud(mesh, self.n_points)
        point_cloud = vertices
        item["point_cloud"] = point_cloud.astype(np.float32)
        item["ious"] = ious
        return item


class RealDatasetMM(Dataset):
    def __init__(self, path, file_name, n_points=256, mode="pc",
                 img_size=128, noise_scale_pc=None, size=None):
        super().__init__()
        self.n_points = n_points
        self.path = path
        self.img_size = img_size
        self.noise_scale_pc = noise_scale_pc
        if mode != "swap":
            self.mode = mode
            self.next_mode = mode
        else:
            self.mode = "pc"
            self.next_mode = "img"
        self.size = size

        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

        if self.size is None:
            self.size = len(self.annotations)

    def swap(self):
        self.mode, self.next_mode = self.next_mode, self.mode

    def __len__(self):
        return min(len(self.annotations), self.size)

    def __getitem__(self, idx):
        item = self.annotations[idx]
        mesh_path = resolve_mesh_path(self.path, item)

        if self.mode == "pc":
            mesh = trimesh.load_mesh(mesh_path)
            mesh = transform_real_mesh(mesh)
            input_item = self.get_point_cloud(mesh)
            input_item["mesh"] = mesh

        elif self.mode == "img":
            input_item = self.get_img(mesh_path, item)
            input_item["mesh"] = None

        elif self.mode == "pc_img":
            if np.random.rand() < 0.5:
                mesh = trimesh.load_mesh(mesh_path)
                mesh = transform_real_mesh(mesh)
                input_item = self.get_point_cloud(mesh)
                input_item["mesh"] = mesh
            else:
                input_item = self.get_img(mesh_path, item)
                input_item["mesh"] = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        input_item["mesh_path"] = mesh_path
        input_item["idx"] = idx
        return input_item

    def get_img(self, mesh_path, item=None):
        images = resolve_images(mesh_path, item)
        return {
            "video": images,
            "description": resolve_description(item if item is not None else {}),
        }

    def get_point_cloud(self, mesh):
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        return {
            "point_cloud": point_cloud,
            "description": "Generate cadquery code",
        }

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(
                loc=0,
                scale=self.noise_scale_pc,
                size=mesh.vertices.shape,
            )
        return mesh


class RealDPODatasetMM(Dataset):
    def __init__(self, path, file_name, tokenizer, n_points=256, mode="pc",
                 img_size=128, noise_scale_pc=None, size=None):
        super().__init__()
        self.n_points = n_points
        self.path = path
        self.img_size = img_size
        self.noise_scale_pc = noise_scale_pc
        self.tokenizer = tokenizer
        if mode != "swap":
            self.mode = mode
            self.next_mode = mode
        else:
            self.mode = "pc"
            self.next_mode = "img"
        self.size = size

        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

        if self.size is None:
            self.size = len(self.annotations)

    def swap(self):
        self.mode, self.next_mode = self.next_mode, self.mode

    def __len__(self):
        return min(len(self.annotations), self.size)

    def __getitem__(self, idx):
        orig_item = self.annotations[idx]
        mesh_path = resolve_mesh_path(self.path, orig_item)

        if self.mode == "pc":
            mesh = trimesh.load_mesh(mesh_path)
            mesh = transform_real_mesh(mesh)
            input_item = self.get_point_cloud(mesh)
            input_item["mesh"] = mesh

        elif self.mode == "img":
            input_item = self.get_img(mesh_path, orig_item)
            input_item["mesh"] = None

        elif self.mode == "pc_img":
            if np.random.rand() < 0.5:
                mesh = trimesh.load_mesh(mesh_path)
                mesh = transform_real_mesh(mesh)
                input_item = self.get_point_cloud(mesh)
                input_item["mesh"] = mesh
            else:
                input_item = self.get_img(mesh_path, orig_item)
                input_item["mesh"] = None
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        with open(os.path.join(self.path, orig_item["ious"]), "rb") as f:
            ious = np.load(f)

        w, l = random.sample(range(len(ious)), 2)
        if ious[l] > ious[w]:
            w, l = l, w

        with open(os.path.join(self.path, orig_item[w]), "r") as f:
            input_item["py_string_w"] = f.read() + self.tokenizer.eos_token
        with open(os.path.join(self.path, orig_item[l]), "r") as f:
            input_item["py_string_l"] = f.read() + self.tokenizer.eos_token

        input_item["mesh_path"] = mesh_path
        input_item["idx"] = idx
        return input_item

    def get_img(self, mesh_path, item=None):
        images = resolve_images(mesh_path, item)
        return {
            "video": images,
            "description": resolve_description(item if item is not None else {}),
        }

    def get_point_cloud(self, mesh):
        mesh = self._augment_pc(mesh)
        point_cloud = mesh_to_point_cloud(mesh, self.n_points)
        return {
            "point_cloud": point_cloud,
            "description": "Generate cadquery code",
        }

    def _augment_pc(self, mesh):
        if self.noise_scale_pc is not None and np.random.rand() < 0.5:
            mesh.vertices += np.random.normal(
                loc=0,
                scale=self.noise_scale_pc,
                size=mesh.vertices.shape,
            )
        return mesh


class Text2CADDataset(Dataset):
    def __init__(self, path, file_name, idx_offset=0, n_samples=None):
        super().__init__()
        self.path = path
        self.n_samples = n_samples
        self.idx_offset = idx_offset
        with open(os.path.join(path, file_name), "rb") as f:
            self.annotations = pickle.load(f)

    def __len__(self):
        return self.n_samples if self.n_samples is not None else len(self.annotations)

    def __getitem__(self, index):
        item = self.annotations[index]

        mesh_path = os.path.join(self.path, item["uid"] + ".stl")
        mesh = trimesh.load_mesh(mesh_path)
        mesh = transform_real_mesh(mesh)

        input_item = {
            "description": item["description"],
            "file_name": item["uid"],
            "mesh": mesh,
            "mesh_path": mesh_path,
            "idx": index + self.idx_offset,
        }
        return input_item


class IndexBuffer:
    def __init__(self, max_size=200):
        self.buffer = deque()
        self.max_size = max_size

    def add(self, index):
        self.buffer.append(index)
        self._enforce_max_size()

    def add_many(self, indices):
        self.buffer.extend(indices)
        self._enforce_max_size()

    def sample(self, n):
        if n > len(self.buffer):
            raise ValueError("Not enough elements in the buffer to sample.")
        return random.sample(self.buffer, n)

    def _enforce_max_size(self):
        if self.max_size is not None:
            while len(self.buffer) > self.max_size:
                self.buffer.popleft()

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"IndexBuffer({list(self.buffer)})"