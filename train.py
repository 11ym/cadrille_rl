import os
import json
from functools import partial
from argparse import ArgumentParser

import torch
from torch.utils.data import ConcatDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from cadrille import Cadrille, collate
from dataset import Text2CADDataset, CadRecodeDataset


class PrintToFileCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            os.makedirs(args.logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs is not None:
            record = {"step": state.global_step, **logs}
            log_file = os.path.join(args.logging_dir, "log.jsonl")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(data_path, log_path, mode, use_text, base_model):
    if not os.path.exists(base_model):
        raise FileNotFoundError(f"本地 base model 路径不存在: {base_model}")

    cad_recode_path = os.path.join(data_path, "cad-recode-v1.5")

    train_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split="train",
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode,
    )

    batch_size = 2
    accumulation_steps = 2

    if use_text:
        text_dataset = Text2CADDataset(
            root_dir=os.path.join(data_path, "text2cad"),
            split="train",
        )

        train_dataset = ConcatDataset([train_dataset, text_dataset])

        batch_size = 2
        accumulation_steps = 8

    eval_dataset = CadRecodeDataset(
        root_dir=cad_recode_path,
        split="val",
        n_points=256,
        normalize_std_pc=100,
        noise_scale_pc=None,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode=mode,
    )

    processor = AutoProcessor.from_pretrained(
        base_model,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        padding_side="left",
        local_files_only=True,
    )

    model = Cadrille.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        local_files_only=True,
    )

    logging_dir = os.path.join(log_path, "runs")

    training_args = TrainingArguments(
        output_dir=log_path,
        logging_dir=logging_dir,

        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=accumulation_steps,

        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=False,

        remove_unused_columns=False,

        max_steps=120000,

        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1000,
        weight_decay=0.01,

        bf16=True,

        logging_steps=100,

        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,

        eval_strategy="steps",
        eval_steps=2000,

        load_best_model_at_end=True,

        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate, processor=processor, n_points=256),
        processing_class=processor,
        callbacks=[PrintToFileCallback()],
    )

    # trainer.train(resume_from_checkpoint=True)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--data-path", type=str, default="./data")
    parser.add_argument("--log-path", type=str, default="./work_dirs")
    parser.add_argument("--mode", type=str, default="pc_img")
    parser.add_argument("--base-model", type=str, default="./models/Qwen2-VL-2B-Instruct")
    parser.add_argument("--use-text", action="store_true")

    args = parser.parse_args()

    run(
        data_path=args.data_path,
        log_path=args.log_path,
        mode=args.mode,
        use_text=args.use_text,
        base_model=args.base_model,
    )