# DeepCAD-only GRPO training entrypoint
# Adapted from the user's original training script.

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
from evaluate import evaluate_model_mm, evaluate_reward_mm


os.environ["PYGLET_HEADLESS"] = "True"
os.environ["TOKENIZERS_PARALLELISM"] = "True"


@dataclass
class TrainConfig:
    sft_path: str

    project: str = "CAD_test"
    group: str = "Dr-CCPO"
    name: str = "dr-ccpo-deepcad-only"
    save_path: str = "./models_notfull"

    failure_reward: float = -10

    train_size: int = None
    # 建议默认先用 pc 跑通；如果 deepcad_train_mesh 里确实有渲染图，再改成 img / pc_img
    train_mode: str = "img"
    train_file: str = "combined_hard_rich_full.pkl"
    freeze_pc: bool = False
    train_epochs: int = 20
    batch_size: int = 16
    save_mid_epoch: bool = True

    epoch_save: int = 1

    temperature: float = 1.0
    do_sample: bool = False
    top_p: float = 1.0

    # GRPO params
    num_generations: int = 16
    top_samples: int = 4
    max_completion_length: int = 400
    learning_rate: float = 3e-5
    batch_updates: int = 3
    epsilon_high: float = 0.1
    epsilon_low: float = 0.1

    use_gpg: bool = False
    use_gspo: bool = False
    use_cov_clip_grpo: bool = False
    use_buffer: bool = False

    num_reward_workers: int = 1
    pool_size: int = 4
    dataloader_workers: int = 4

    iou_coef: int = 10
    cd_coef: int = 0
    auc_coef: int = 0

    processor_path: str = "/mengyiming/cadrille/models/Qwen2-VL-2B-Instruct"
    deepcad_train_path: str = "/mengyiming/cadrille/data/deepcad_train_mesh"
    deepcad_test_path: str = "/mengyiming/cadrille/data/deepcad_test_mesh"
    eval_size: int = 1000
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

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

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
        return_tensors="pt"
    )

    inputs['point_clouds'] = torch.stack([
        torch.tensor(m['point_cloud']) if is_pc[i] else torch.zeros(n_points, 3)
        for i, m in enumerate(batch)
    ])
    inputs['is_pc'] = torch.tensor(is_pc, dtype=torch.bool)
    inputs['is_img'] = torch.tensor(is_img, dtype=torch.bool)

    if 'pixel_values_videos' in inputs.keys():
        pixel_values_videos = inputs['pixel_values_videos'].new_zeros(
            (len(batch), torch.prod(inputs['video_grid_thw'][0]), inputs['pixel_values_videos'].shape[1])
        )
        pixel_values_videos[inputs['is_img']] = torch.stack(
            torch.chunk(inputs['pixel_values_videos'], chunks=sum(inputs['is_img']))
        )
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
        rewards = []
        pred_metrics = get_metrics_from_texts(completions, answer, max_workers=23)
        for m in pred_metrics:
            reward = 0
            iou = m["iou"] if m is not None else None
            cd = m["cd"] if m is not None else None
            if iou is None:
                reward = failure_reward
            else:
                reward = iou * iou_coef + np.clip(1 - cd * 1000, 0, 1) * cd_coef
            rewards.append(reward)
        return rewards
    return combined_reward



def optimize_model_memory(model):
    model.train()
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.gradient_checkpointing_enable()
    return model



def setup(world_size):
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        timeout=timedelta(hours=5)
    )



def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@record
@pyrallis.wrap()
def main(config: TrainConfig):
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    setup(world_size)

    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(rank)
    print("RANK, WS:", rank, world_size, flush=True)

    model = Cadrille.from_pretrained(
        config.sft_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=rank,
    ).train().to(rank)

    processor = AutoProcessor.from_pretrained(
        config.processor_path,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left"
    )

    eval_data_deepcad = RealDatasetMM(
        path=config.deepcad_test_path,
        file_name='test.pkl',
        n_points=256,
        size=config.eval_size,
        mode='pc',
    )

    train_data = RealDatasetMM(
        path=config.deepcad_train_path,
        file_name=config.train_file,
        n_points=256,
        mode=config.train_mode,
        noise_scale_pc=0.01,
        size=config.train_size,
    )

    print(f"Train dataset size: {len(train_data)}", flush=True)
    print(f"Eval dataset size: {len(eval_data_deepcad)}", flush=True)
    print(f"Detected {torch.cuda.device_count()} GPUs", flush=True)

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
        n_points=256,
    )

    ious, cds, incorrect, ious_im, cds_im, incorrect_im = np.zeros(6, dtype=np.float32)
    initial_reward_pc = None
    initial_reward_img = None

    if rank == 0 and config.initial_eval:
        print("\nInitial SFT evaluation before RL fine-tuning:", flush=True)
        eval_data_deepcad.mode = "pc"
        ious, cds, incorrect, _ = evaluate_model_mm(
            config, model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200
        )
        eval_data_deepcad.mode = "img"
        ious_im, cds_im, incorrect_im, _ = evaluate_model_mm(
            config, model.module, processor, eval_data_deepcad, rank, part_collate, batch_size=200
        )

        reward_function = get_reward_function(
            config.failure_reward,
            iou_coef=config.iou_coef,
            auc_coef=config.auc_coef,
            cd_coef=config.cd_coef,
        )
        eval_data_deepcad.mode = "pc"
        initial_reward_pc = evaluate_reward_mm(
            config,
            model.module,
            processor,
            eval_data_deepcad,
            rank,
            part_collate,
            reward_function,
            batch_size=config.reward_eval_batch_size,
            num_generations=config.reward_eval_generations,
            max_eval_samples=config.reward_eval_size,
        )
        eval_data_deepcad.mode = "img"
        initial_reward_img = evaluate_reward_mm(
            config,
            model.module,
            processor,
            eval_data_deepcad,
            rank,
            part_collate,
            reward_function,
            batch_size=config.reward_eval_batch_size,
            num_generations=config.reward_eval_generations,
            max_eval_samples=config.reward_eval_size,
        )
    dist.barrier()

    print("\nStarting RL fine-tuning using GRPO (DeepCAD-only)...", flush=True)
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
    }

    sampler = DistributedSampler(
        train_data,
        num_replicas=world_size,
        rank=rank,
    )

    run_id = None
    if rank == 0:
        dict_config = asdict(config)
        wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            reinit=True,
            config=dict_config,
        )
        print("Weights & Biases initialized.", flush=True)
        run_id = wandb.run.id

        initial_logs = {
            "eval/pc/DeepCAD test/IoU mean": np.mean(ious),
            "eval/pc/DeepCAD test/CD mean": np.mean(cds),
            "eval/pc/DeepCAD test/IoU median": np.median(ious),
            "eval/pc/DeepCAD test/CD median": np.median(cds),
            "eval/pc/DeepCAD test/Failures fraction": incorrect,
            "eval/img/DeepCAD test/IoU mean": np.mean(ious_im),
            "eval/img/DeepCAD test/CD mean": np.mean(cds_im),
            "eval/img/DeepCAD test/IoU median": np.median(ious_im),
            "eval/img/DeepCAD test/CD median": np.median(cds_im),
            "eval/img/DeepCAD test/Failures fraction": incorrect_im,
        }
        if initial_reward_pc is not None:
            initial_logs.update({
                "reward_eval/pc/DeepCAD/mean": initial_reward_pc["reward_mean"],
                "reward_eval/pc/DeepCAD/median": initial_reward_pc["reward_median"],
                "reward_eval/pc/DeepCAD/std": initial_reward_pc["reward_std"],
                "reward_eval/pc/DeepCAD/failure_ratio": initial_reward_pc["failure_ratio"],
                "reward_eval/pc/DeepCAD/num_samples": initial_reward_pc["num_reward_samples"],
            })
        if initial_reward_img is not None:
            initial_logs.update({
                "reward_eval/img/DeepCAD/mean": initial_reward_img["reward_mean"],
                "reward_eval/img/DeepCAD/median": initial_reward_img["reward_median"],
                "reward_eval/img/DeepCAD/std": initial_reward_img["reward_std"],
                "reward_eval/img/DeepCAD/failure_ratio": initial_reward_img["failure_ratio"],
                "reward_eval/img/DeepCAD/num_samples": initial_reward_img["num_reward_samples"],
            })
        wandb.log(initial_logs)

    reward_function = get_reward_function(
        config.failure_reward,
        iou_coef=config.iou_coef,
        auc_coef=config.auc_coef,
        cd_coef=config.cd_coef,
    )

    model = train_with_grpo_mm(
        model=model,
        processor=processor,
        train_data=train_data,
        eval_data_deepcad=eval_data_deepcad,
        eval_data_fusion=None,
        eval_data_text=None,
        sampler=sampler,
        reward_function=reward_function,
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
        **training_config,
    )

    if rank == 0:
        wandb.finish()
        print("Training completed and wandb run finished.", flush=True)
        print("\nSaving GRPO fine-tuned model...", flush=True)
        model.save_pretrained(f"{config.save_path}/{run_id}")
        processor.save_pretrained(f"{config.save_path}/{run_id}")

    cleanup()


if __name__ == "__main__":
    main()
