# CAD recode imports
import multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from qwen_vl_utils import process_vision_info
from torch.utils.data import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import torch
import torch.distributed as dist
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from functools import partial

import pyrallis
import wandb
from cad_recode_model_mm import Cadrille
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import AutoProcessor

from dataset_utils import RealDatasetMM
from grpo_mm import train_with_grpo_mm
from utils_cadrille import get_metrics_from_texts


os.environ["PYGLET_HEADLESS"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"


@dataclass
class TrainConfig:
    sft_path: str

    project: str = "CAD_test"
    group: str = "Dr-CCPO"
    name: str = "dr-ccpo-run2"
    save_path: str = "./models_notfull_run2"

    failure_reward: float = -10

    train_size: int = None
    train_mode: str = "img"
    # train_file: str = "combined_hard.pkl"
    train_file: str = "combined_hard_rich_light.pkl"
    freeze_pc: bool = False
    train_epochs: int = 20
    batch_size: int = 16  # 4 A100: 4 per GPU (降低以避免 OOM)
    save_mid_epoch: bool = True
    test_save_steps: int = None  # 如果设置，跑这么多步后就保存并退出（用于测试）

    epoch_save: int = 1

    temperature: float = 0.9  # 0.8-1.0 for reasoning tasks
    do_sample: bool = True  # MUST be True for GRPO
    top_p: float = 0.95
    # GRPO params - 降低以节省显存
    num_generations: int = 12  # 从 16 降到 12
    top_samples: int = 6  # 选择 2/3 用于训练
    max_completion_length: int = 250  # 从 250 降到 200
    learning_rate: float = 1e-5  # 按 batch_size 比例调整 (论文 3e-5 @ batch=128)
    batch_updates: int = 3  # 论文配置: updates per batch
    # clipping epsilon values - 论文配置 PPO ε=0.1
    epsilon_high: float = 0.1  # 论文标准配置
    epsilon_low: float = 0.1  # 论文标准配置
    kl_coef: float = 0.02  # KL divergence penalty coefficient (β) - standard config
    kl_target_low: float = 0.1  # If KL < 0.1, reduce β to 0.01
    kl_target_high: float = 0.5  # If KL > 0.5, increase β to 0.05

    use_gpg: bool = False
    use_gspo: bool = False
    use_cov_clip_grpo: bool = False
    use_buffer: bool = False

    num_reward_workers : int = 8
    pool_size : int = 8
    dataloader_workers : int = 8

    # reward params
    iou_coef : int = 10
    cd_coef : int = 0
    auc_coef : int = 0
    initial_eval: bool = True
    full_eval_interval_epochs: int = 1
    reward_eval_interval_steps: int = 50
    reward_eval_size: int = 128
    reward_eval_batch_size: int = 32
    reward_eval_generations: int = 1
    quick_eval_interval_steps: int = 100
    quick_eval_size: int = 200
    quick_eval_batch_size: int = 64


def collate_img_pc_v1(batch, processor, n_points, eval=False):
    messages = []
    is_pc = [0] * len(batch)
    is_img = [0] * len(batch)

    for i, m in enumerate(batch):
        if 'video' in m.keys():
            is_img[i] = 1
            message = [{
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': m['video'], 'fps': 1.0},
                    {'type': 'text', 'text': m['description']}
                ]
            }]
        else:
            if 'point_cloud' in m.keys():
                is_pc[i] = 1
            message = [{
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': m['description']}
                ]
            }]
        messages.append(message)
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
             for msg in messages]

    points_inputs = ''.join(n_points * [processor.tokenizer.pad_token])

    for i in range(len(texts)):
        if is_pc[i]:
            texts[i] = points_inputs + texts[i]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt")

    inputs['point_clouds'] = torch.stack([torch.tensor(m['point_cloud']) if is_pc[i]
                                          else torch.zeros(n_points, 3) for i, m in enumerate(batch)])
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros(
            (len(batch), torch.prod(inputs['video_grid_thw'][0]),
             inputs['pixel_values_videos'].shape[1]))
        pixel_values_videos[inputs['is_img']] = torch.stack(torch.chunk(inputs['pixel_values_videos'],
                                                                        chunks=sum(inputs['is_img'])))
        inputs['pixel_values_videos'] = pixel_values_videos

        video_grid_thw = inputs['video_grid_thw'].new_zeros((len(batch), 3))
        video_grid_thw[inputs['is_img']] = inputs['video_grid_thw']
        inputs['video_grid_thw'] = video_grid_thw

    inputs['mesh_path'] = [m['mesh_path'] for m in batch]
    inputs['mesh'] = [m['mesh'] for m in batch]
    inputs['idx'] = [m['idx'] for m in batch]
    return inputs


def get_reward_function(failure_reward, iou_coef=10, auc_coef=0, cd_coef=0):
    def combined_reward(completions, answer):
        torch.cuda.synchronize()
        # Get individual rewards
        rewards = []
        # excepts = []
        pred_metrics = get_metrics_from_texts(
            completions, answer, max_workers=8)  # Reduced for 2 GPUs
        # print("MESHES", pred_meshes, flush=True)
        for m in pred_metrics:
            reward = 0
            iou = m["iou"] if m is not None else None
            #auc =  m["auc"] if m is not None else None
            cd =  m["cd"] if m is not None else None
            if iou is None:
                reward = failure_reward
            #elif iou < 0:
            #    reward = 0
            else:
                #print(f"Chamfer Distance: {cd}")
                #reward = np.clip(1 - cd * 5, 0, 1) * cd_coef
                #reward = np.clip(-1/6 * np.log10(cd), 0, 1) * cd_coef
                reward = iou * iou_coef + np.clip(1 - cd * 1000, 0, 1) * cd_coef
            rewards.append(reward)
        return rewards
    return combined_reward


def optimize_model_memory(model):
    """
    Optimizes the model to use less memory during training.
    """
    model.train()
    model.config.use_cache = False

    # First ensure inputs will require gradients
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # Then enable gradient checkpointing
    model.gradient_checkpointing_enable()

    return model


def setup(world_size):
    """ Initialize the process group for distributed training """
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        timeout=timedelta(
            hours=5))


def cleanup():
    """ Destroy the process group """
    dist.destroy_process_group()


@record
@pyrallis.wrap()
def main(config: TrainConfig):
    world_size = int(os.getenv("WORLD_SIZE"))
    setup(world_size)

    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()

    torch.cuda.set_device(rank)
    print("RANK, WS:", rank, world_size, flush=True)
    attn_implementation = 'flash_attention_2' if torch.cuda.is_available() else None

    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=rank).train().to(rank)

    processor = AutoProcessor.from_pretrained("/mengyiming/cadrille/models/Qwen2-VL-2B-Instruct",
                                              min_pixels=256 * 28 * 28,
                                              max_pixels=1280 * 28 * 28,
                                              padding_side="left")

    eval_data_deepcad = RealDatasetMM(
        path=f'/mengyiming/cadrille/data/deepcad_test_mesh',
        file_name='test.pkl',
        n_points=256,
        size=1000)
    eval_data_fusion = RealDatasetMM(
        path=f'/mengyiming/cadrille/data/fusion360_test_mesh',
        file_name='test.pkl',
        n_points=256,
        size=1000)
    train_data = RealDatasetMM(
        path=f'/mengyiming/cadrille/data/deepcad_fusion_train',
        file_name=config.train_file,
        n_points=256,
        mode=config.train_mode,
        noise_scale_pc=0.01,
        size=config.train_size)

    # text_train_dataset = Text2CADDataset(path=f'/home/jovyan/tarasov/data/deepcad_fusion_train', file_name='text_train.pkl', idx_offset=len(train_data))
    # text_eval_dataset = Text2CADDataset(path=f'/home/jovyan/tarasov/data/deepcad_test', file_name='text_test.pkl')
    # train_data = ConcatDataset([train_data, text_train_dataset])

    # Main execution
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPUs")

    model = optimize_model_memory(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    if config.freeze_pc:
        model.freeze_pc()

    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    dist.barrier()
    part_collate = partial(
        collate_img_pc_v1,
        processor=processor,
        n_points=256)

    """
    if rank == 0:
        print("\nInitial model evaluation before finetuning and after filtering:")
        eval_data_deepcad.mode = 'pc'
        ious, cds, incorrect, failed_intersect = evaluate_model_mm(model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200)
        eval_data_deepcad.mode = 'img'
        ious_im, cds_im, incorrect_im, failed_intersect_im = evaluate_model_mm(model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200)
        eval_data_fusion.mode = 'pc'
        ious_f, cds_f, incorrect_f, failed_intersect_f = evaluate_model_mm(model.module, processor, eval_data_fusion, rank, part_collate, batch_size=200)
        eval_data_fusion.mode = 'img'
        ious_f_im, cds_f_im, incorrect_f_im, failed_intersect_f_im = evaluate_model_mm(model.module, processor, eval_data_fusion, rank, part_collate, batch_size=200)

        # ious_txt, cds_txt, incorrect_txt, failed_intersect_txt = evaluate_model_mm(model.module, processor, text_eval_dataset, rank, part_collate, batch_size=50)"""

    ious, cds, incorrect, ious_f, cds_f, incorrect_f, ious_im, cds_im, incorrect_im, ious_f_im, cds_f_im, incorrect_f_im = np.zeros(
        12, dtype=np.float32)
    dist.barrier()

    print("\nStarting RL fine-tuning using GRPO...")
    training_config = {
        'train_epochs': config.train_epochs,
        'batch_size': config.batch_size,
        'num_generations': config.num_generations,
        'top_samples': config.top_samples,
        'max_completion_length': config.max_completion_length,
        'learning_rate': config.learning_rate,
        'batch_updates': config.batch_updates,
        'epsilon_high': config.epsilon_high,
        'epsilon_low': config.epsilon_low,
        'kl_coef': config.kl_coef,
        'kl_target_low': config.kl_target_low,
        'kl_target_high': config.kl_target_high,
        'num_workers': config.dataloader_workers,
        'temperature': config.temperature,
        'do_sample': config.do_sample,
        'top_p': config.top_p,
        'test_save_steps': config.test_save_steps,
    }
    sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank)
    # Initialize Weights & Biases
    run_id = None
    if rank == 0:
        dict_config = asdict(config)
        import time
        for attempt in range(3):
            try:
                wandb.init(
                    project=config.project,
                    group=config.group,
                    name=config.name,
                    reinit=True,
                    config=dict_config)
                print("Weights & Biases initialized.")
                break
            except Exception as e:
                print(f"wandb init attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5)
                else:
                    print("wandb init failed after 3 attempts, continuing without wandb")
        run_id = wandb.run.id if wandb.run else "no_wandb"

        wandb.log({
            "eval/pc/DeepCAD test/IoU mean": np.mean(ious),
            "eval/pc/DeepCAD test/CD mean": np.mean(cds),
            "eval/pc/DeepCAD test/IoU median": np.median(ious),
            "eval/pc/DeepCAD test/CD median": np.median(cds),
            "eval/pc/DeepCAD test/Failures fraction": incorrect,
            "eval/pc/Fusion360 test/IoU mean": np.mean(ious_f),
            "eval/pc/Fusion360 test/CD mean": np.mean(cds_f),
            "eval/pc/Fusion360 test/IoU median": np.median(ious_f),
            "eval/pc/Fusion360 test/CD median": np.median(cds_f),
            "eval/pc/Fusion360 test/Failures fraction": incorrect_f,

            "eval/img/DeepCAD test/IoU mean": np.mean(ious_im),
            "eval/img/DeepCAD test/CD mean": np.mean(cds_im),
            "eval/img/DeepCAD test/IoU median": np.median(ious_im),
            "eval/img/DeepCAD test/CD median": np.median(cds_im),
            "eval/img/DeepCAD test/Failures fraction": incorrect_im,
            "eval/img/Fusion360 test/IoU mean": np.mean(ious_f_im),
            "eval/img/Fusion360 test/CD mean": np.mean(cds_f_im),
            "eval/img/Fusion360 test/IoU median": np.median(ious_f_im),
            "eval/img/Fusion360 test/CD median": np.median(cds_f_im),
            "eval/img/Fusion360 test/Failures fraction": incorrect_f_im,

            # "eval/txt/DeepCAD test/IoU mean": np.mean(ious_txt),
            # "eval/txt/DeepCAD test/CD mean": np.mean(cds_txt),
            # "eval/txt/DeepCAD test/IoU median": np.median(ious_txt),
            # "eval/txt/DeepCAD test/CD median": np.median(cds_txt),
            # "eval/txt/DeepCAD test/Failures fraction": incorrect_txt + failed_intersect_txt,
        })

    model = train_with_grpo_mm(
        model=model,
        processor=processor,
        train_data=train_data,
        eval_data_deepcad=eval_data_deepcad,
        eval_data_fusion=eval_data_fusion,
        # eval_data_text=text_eval_dataset,
        eval_data_text=None,
        sampler=sampler,
        reward_function=get_reward_function(
            config.failure_reward,
            iou_coef=config.iou_coef,
            cd_coef=config.cd_coef,
            auc_coef=config.auc_coef,
        ),
        collate_fn=part_collate,
        run_id=run_id,
        gpg=config.use_gpg,
        use_buffer=config.use_buffer,
        save_path=config.save_path,
        config=config,
        reward_eval_interval_steps=config.reward_eval_interval_steps,
        reward_eval_size=config.reward_eval_size,
        reward_eval_batch_size=config.reward_eval_batch_size,
        reward_eval_generations=config.reward_eval_generations,
        quick_eval_interval_steps=config.quick_eval_interval_steps,
        quick_eval_size=config.quick_eval_size,
        quick_eval_batch_size=config.quick_eval_batch_size,
        full_eval_interval_epochs=config.full_eval_interval_epochs,
        **training_config
    )
    if rank == 0:

        wandb.finish()
        print("Training completed and wandb run finished.")

        print("\nSaving GRPO fine-tuned model...")

        model.save_pretrained(f"{config.save_path}/{run_id}")
        processor.save_pretrained(f"{config.save_path}/{run_id}")
    cleanup()


if __name__ == "__main__":
    main()
