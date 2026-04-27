import os

import numpy as np
import wandb
import time

os.environ["PYGLET_HEADLESS"] = "True"

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
import torch.nn.functional as F

from evaluate import evaluate_model_mm, evaluate_reward_mm
from dataset_utils import IndexBuffer


def selective_log_softmax(logits, input_ids):
    """
    Computes log probabilities for specific tokens in the vocabulary.
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)


def compute_log_probs(model, batch, logits_to_keep):
    """
    Computes the log probabilities for a batch of tokens.
    """
    input_ids, attention_mask, point_cloud, is_pc, is_img, pixel_values_videos, video_grid_thw = batch
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.to(model.device)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.to(model.device)
    logits = model(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        point_clouds=point_cloud.clone(),
        is_pc=is_pc.to(model.device),
        is_img=is_img.to(model.device),
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw).logits[:, :-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, input_ids)


def create_completion_mask(completion_ids, eos_token_id):
    """
    Creates a mask for completion tokens that excludes tokens after the EOS token.
    """
    is_eos = completion_ids == eos_token_id
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    mask_exists = is_eos.any(dim=1)
    eos_idx[mask_exists] = is_eos.int().argmax(dim=1)[mask_exists]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).int()


def validate_generation_inputs(model, prompt_ids, prompt_mask, processor):
    vocab_size = model.config.vocab_size
    if prompt_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"input_ids must be integer typed, got {prompt_ids.dtype}")

    bad_token_mask = (prompt_ids < 0) | (prompt_ids >= vocab_size)
    if bad_token_mask.any():
        bad_tokens = prompt_ids[bad_token_mask][:20].detach().cpu().tolist()
        raise ValueError(
            f"Found invalid token ids before generation. "
            f"Valid range is [0, {vocab_size - 1}], examples={bad_tokens}"
        )

    if prompt_mask.shape != prompt_ids.shape:
        raise ValueError(
            f"attention_mask shape {prompt_mask.shape} does not match input_ids shape {prompt_ids.shape}"
        )

    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is not None and (eos_token_id < 0 or eos_token_id >= vocab_size):
        raise ValueError(f"Invalid eos_token_id={eos_token_id} for vocab_size={vocab_size}")


def get_valid_bad_words_ids(model):
    vocab_size = model.config.vocab_size
    bad_words_ids = []
    for attr in ["image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id"]:
        token_id = getattr(model.config, attr, None)
        if isinstance(token_id, int) and 0 <= token_id < vocab_size:
            bad_words_ids.append([token_id])
    return bad_words_ids


def generate_completions(model, processor, inputs, num_generations=4, max_completion_length=32, temperature=1.0, top_p=1.0, do_sample=False):
    """
    Generates multiple completions for each prompt.
    """
    device = model.device
    prompt_ids = inputs["input_ids"].clone().detach().to(device)
    prompt_mask = inputs["attention_mask"].clone().detach().to(device)
    point_cloud = inputs["point_clouds"].clone().detach().to(device)
    is_pc = inputs["is_pc"].clone().detach().to(device)
    is_img = inputs["is_img"].clone().detach().to(device)
    pixel_values_videos = inputs['pixel_values_videos'].clone().detach().to(device) if inputs.get('pixel_values_videos',
                                                                                               None) is not None else None
    video_grid_thw = inputs['video_grid_thw'].clone().detach().to(device) if inputs.get('video_grid_thw',
                                                                                     None) is not None else None
    prompt_length = prompt_ids.size(1)
    batch_size = prompt_ids.size(0)
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)
    point_cloud = point_cloud.repeat_interleave(num_generations, dim=0)
    is_pc = is_pc.repeat_interleave(num_generations, dim=0)
    is_img = is_img.repeat_interleave(num_generations, dim=0)
    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.repeat_interleave(num_generations, dim=0)
    if video_grid_thw is not None:
        video_grid_thw = video_grid_thw.repeat_interleave(num_generations, dim=0)

    validate_generation_inputs(model, prompt_ids, prompt_mask, processor)
    bad_words_ids = get_valid_bad_words_ids(model)

    print(f"Generating {num_generations} completions for each of {batch_size} prompts.", flush=True)
    outputs = model.generate(input_ids=prompt_ids.clone(),
                             attention_mask=prompt_mask.clone(),
                             point_clouds=point_cloud.clone(),
                             is_pc=is_pc.clone(),
                             is_img=is_img.clone(),
                             pixel_values_videos=pixel_values_videos.clone() if pixel_values_videos is not None else None,
                             video_grid_thw=video_grid_thw.clone() if video_grid_thw is not None else None,
                             max_new_tokens=max_completion_length,
                             do_sample=do_sample,
                             temperature=temperature,
                             top_p=top_p,
                             top_k=50,
                             early_stopping=False,
                             bad_words_ids=bad_words_ids if bad_words_ids else None
                             )
    completion_ids = outputs[:, prompt_length:]
    completion_mask = create_completion_mask(completion_ids, processor.tokenizer.eos_token_id)

    # Debug: Check diversity (only on rank 0)
    is_main = (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    if is_main:
        print(f"[DEBUG] Sampling: do_sample={do_sample}, temp={temperature}, top_p={top_p}", flush=True)
        for i in range(min(batch_size, 2)):
            start_idx = i * num_generations
            end_idx = start_idx + num_generations
            batch_completions = completion_ids[start_idx:end_idx]
            batch_masks = completion_mask[start_idx:end_idx]

            normed = []
            for c, m in zip(batch_completions, batch_masks):
                valid = c[m.bool()].tolist()
                normed.append(tuple(valid))

            unique_count = len(set(normed))
            print(f"[DEBUG] Prompt {i}: {unique_count}/{num_generations} unique completions", flush=True)

    return point_cloud, prompt_ids, prompt_mask, is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask


def generate_rollout_data(model, reward_function,
                          processor, batch_samples, num_generations, max_completion_length, top_samples=None,
                          gpg=False, buffer=None, temperature=1.0, top_p=1.0, do_sample=False):
    """
    Generates data for GRPO rollouts including completions and log probabilities.
    """
    prompts = batch_samples
    with torch.no_grad():
        t0 = time.perf_counter()
        point_cloud, prompt_ids, prompt_mask, is_pc, is_img, pixel_values_videos, video_grid_thw, completion_ids, completion_mask = generate_completions(
        model=model,
        processor=processor,
        inputs=prompts,
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,

        )

        gen_time = time.perf_counter() - t0
        print(f"[TIME] generation time: {gen_time:.3f} s", flush=True)

        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        formatted_completions = [processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False) for ids in completion_ids]
        repeated_answers = [a for a in batch_samples['mesh_path'] for _ in range(num_generations)]
        #print("getting rewards", flush=True)

        t1 = time.perf_counter()

        rewards = torch.tensor(
            reward_function(completions=formatted_completions, answer=repeated_answers),
            dtype=torch.float32,
            device=model.device
        )
        reward_time = time.perf_counter() - t1
        print(f"[TIME] reward computation time: {reward_time:.3f} s", flush=True)
        print("Rewards", rewards, flush=True)

        batch_size = len(prompts['input_ids'])
        num_generations = num_generations
        if top_samples is None:
            top_samples = num_generations
        rewards = rewards.view(batch_size, num_generations)
        avg_reward = rewards.mean().item()
        reward_std = rewards.std().item()
        invalid_ratio = (rewards == -10).float().mean().item()

        # Count unique completions
        unique_completions = len(set(formatted_completions))

        print("Average Reward:", avg_reward, flush=True)
        print(f"Reward std: {reward_std:.4f}, Invalid ratio: {invalid_ratio:.4f}, Unique: {unique_completions}/{len(formatted_completions)}", flush=True)

        stats = {
            "avg_reward": avg_reward,
            "reward_std": reward_std,
            "invalid_ratio": invalid_ratio,
            "unique_completions": unique_completions,
            "total_completions": len(formatted_completions)
        }

        # Filter out samples with low reward variance (no learning signal)
        reward_var_per_sample = rewards.var(dim=1)
        valid_mask = reward_var_per_sample >= 0.01
        num_filtered = (~valid_mask).sum().item()

        if num_filtered > 0:
            print(f"[FILTER] Skipping {num_filtered}/{batch_size} samples with reward variance < 0.01", flush=True)

        if valid_mask.sum() == 0:
            print("[FILTER] All samples filtered out, skipping this batch", flush=True)
            return None, stats

        # Only keep valid samples
        rewards = rewards[valid_mask]
        batch_size = rewards.shape[0]

        # Filter all tensors to match valid samples
        valid_indices = torch.arange(len(valid_mask), device=model.device)[valid_mask]
        filter_indices = valid_indices.unsqueeze(1) * num_generations + torch.arange(num_generations, device=model.device)
        filter_indices = filter_indices.view(-1)

        input_ids = input_ids[filter_indices]
        attention_mask = attention_mask[filter_indices]
        point_cloud = point_cloud[filter_indices]
        completion_mask = completion_mask[filter_indices]
        is_pc = is_pc[filter_indices]
        is_img = is_img[filter_indices]
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos[filter_indices]
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw[filter_indices]

        mean_rewards = rewards.mean(dim=1).repeat_interleave(num_generations)

        abs_adv = torch.abs(rewards - mean_rewards.view(batch_size, num_generations))
        #adv = rewards - mean_rewards.view(batch_size, num_generations)
        # gets the indices of the top samples based on absolute advantages

        _, top_indices = torch.topk(abs_adv, top_samples, dim=1)
        #_, top_indices = torch.topk(adv, top_samples, dim=1)
        #print("NOT using absolute advantages for GRPO loss")


        row_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, top_samples).to(model.device)
        flattened_indices = row_indices * num_generations + top_indices

        # Compute advantages in grouped shape first so normalization is done within each prompt group.
        raw_advantages = rewards.view(-1) - mean_rewards
        advantages = raw_advantages[flattened_indices].reshape(batch_size, top_samples)

        # Group-normalize advantages to stay faithful to GRPO.
        group_adv_mean = advantages.mean(dim=1, keepdim=True)
        group_adv_std = advantages.std(dim=1, keepdim=True)
        advantages = (advantages - group_adv_mean) / (group_adv_std + 1e-8)
        advantages = advantages.reshape(batch_size * top_samples, 1)

        input_ids = input_ids[flattened_indices].reshape(batch_size * top_samples, *input_ids.shape[1:])
        attention_mask = attention_mask[flattened_indices].reshape(batch_size * top_samples, *attention_mask.shape[1:])
        point_cloud = point_cloud[flattened_indices].reshape(batch_size * top_samples, *point_cloud.shape[1:])
        completion_mask = completion_mask[flattened_indices].reshape(batch_size * top_samples,
                                                                     *completion_mask.shape[1:])
        is_pc = is_pc[flattened_indices].reshape(batch_size * top_samples, *is_pc.shape[1:])
        is_img = is_img[flattened_indices].reshape(batch_size * top_samples, *is_img.shape[1:])
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos[flattened_indices].reshape(batch_size * top_samples,
                                                                       *pixel_values_videos.shape[1:])
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw[flattened_indices].reshape(batch_size * top_samples, *video_grid_thw.shape[1:])
        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "formatted_completions": formatted_completions,
            "repeated_answers": repeated_answers,
            "logits_to_keep": logits_to_keep,
            "batch_size": len(prompts['input_ids']),
            "num_generations": num_generations,
            "point_cloud": point_cloud,
            "advantages": advantages,
            "is_pc": is_pc,
            "is_img": is_img,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }
        if not gpg:
            old_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos.clone() if pixel_values_videos is not None else None, video_grid_thw.clone() if video_grid_thw is not None else None),
                                              logits_to_keep)
            result["old_log_probs"] = old_log_probs.detach()

    return result, stats


def grpo_loss(model, rollout_data, processor, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None, logger=None, kl_coef=0.01):
    """
    Computes the GRPO loss for updating the policy model.
    """
    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)
    ratio_mean = ratio.mean().item()
    print(f"ratio mean {ratio_mean}")
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    per_token_loss = surrogate_loss

    # Compute entropy
    ent_per_token = -token_log_probs.detach()
    ent_per_token = ent_per_token * completion_mask
    seq_entropy = ent_per_token.sum(dim=1) / (completion_mask.sum(dim=1) + 1e-12)
    avg_entropy = seq_entropy.mean().item()

    # Compute KL divergence. Keep a tensor version for optimization and a scalar for logging.
    kl_per_token = (old_log_probs - token_log_probs) * completion_mask
    seq_kl = kl_per_token.sum(dim=1) / (completion_mask.sum(dim=1) + 1e-12)
    kl_loss = seq_kl.mean()
    avg_kl = kl_loss.detach().item()

    # Compute approx KL (ratio-based)
    with torch.no_grad():
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

    if logger :
        covs = (token_log_probs - token_log_probs.mean()) * (advantages - advantages.mean())
        logger.log_metrics({
            "avg_entropy" : avg_entropy,
            "cov_mean": {covs.mean().item()},
            "covs_std": {covs.std().item()},
        })

    # Compute policy loss
    policy_loss = -torch.clamp(torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0), min=-15, max=15).mean()

    # Add KL penalty: loss = policy_loss + β * kl_divergence
    loss = policy_loss + kl_coef * kl_loss

    return loss, avg_entropy, avg_kl, approx_kl, ratio_mean


def grpo_loss_clip_cov(model, rollout_data, processor, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None, logger=None):
    """
    Computes the GRPO loss for updating the policy model.
    """
    cov_lb = 1
    cov_hb = 5
    select_ratio = 2e-4

    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)
    ratio = torch.exp(token_log_probs - old_log_probs)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages

    ### detach top covariance tokens
    covs = (token_log_probs - token_log_probs.mean()) * (advantages - advantages.mean())
    mask = (covs > cov_lb) & (covs < cov_hb)
    all_idx = torch.nonzero(mask).reshape(-1)
    select_num = int(select_ratio * token_log_probs.numel())

    if all_idx.numel() >= select_num > 0:
        perm= torch.randperm(all_idx.numel(), device=all_idx.device)
        clip_idx = all_idx[perm[:select_num]]
        surr1[clip_idx] = surr1[clip_idx].detach()
        surr2[clip_idx] = surr2[clip_idx].detach()

    
    surrogate_loss = torch.min(surr1, surr2)


    per_token_loss = surrogate_loss

    if logger :
        ent_per_token = -token_log_probs.detach()
        ent_per_token = ent_per_token * completion_mask
        seq_entropy = ent_per_token.sum(dim=1) / (completion_mask.sum(dim=1) + 1e-12)
        avg_entropy = seq_entropy.mean().item()
        logger.log_metrics({
            "avg_entropy" : avg_entropy,
            "cov_mean": covs.mean().item(),
            "covs_std": covs.std().item(),
            "select_num" : select_num
        })

        
    loss = -torch.clamp(torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0), min=-15, max=15).mean()
    return loss


def gpg_loss(model, rollout_data, tokenizer, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None):
    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)
    per_token_loss = token_log_probs * advantages
    loss = -torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0).mean()
    return loss


def gspo_loss(model, rollout_data, processor, reward_function, epsilon_high=0.2, epsilon_low=0.2, top_samples=None):
    """
    Computes the GRPO loss for updating the policy model. GSPO update source : TRL by Hugging Face
    """
    device = model.device
    input_ids = rollout_data["input_ids"]
    point_cloud = rollout_data["point_cloud"]
    attention_mask = rollout_data["attention_mask"]
    completion_mask = rollout_data["completion_mask"]
    logits_to_keep = rollout_data["logits_to_keep"]
    old_log_probs = rollout_data["old_log_probs"]
    advantages = rollout_data["advantages"]
    is_pc = rollout_data["is_pc"]
    is_img = rollout_data["is_img"]
    pixel_values_videos = rollout_data["pixel_values_videos"]
    video_grid_thw = rollout_data["video_grid_thw"]
    token_log_probs = compute_log_probs(model, (input_ids.clone(), attention_mask.clone(), point_cloud.clone(), is_pc.clone(), is_img.clone(), pixel_values_videos, video_grid_thw), logits_to_keep)

    log_ratio = token_log_probs - old_log_probs
    log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1)
    log_importance_weights = log_importance_weights.unsqueeze(-1)
    #log_importance_weights : (B, 1)
    ratio = torch.exp(log_importance_weights)

    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * advantages
    surrogate_loss = torch.min(surr1, surr2)
    per_token_loss = surrogate_loss
    loss = -torch.clamp(torch.nan_to_num(((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)), 0, 0,
                             0), min=-15, max=15).mean()
    return loss
    

def merge_collated_batches(batch1, batch2, padding_value):
    merged = {}
    bs1 = batch1['input_ids'].shape[0]
    bs2 = batch2['input_ids'].shape[0]
    for key in batch1:
        if key not in batch2:
            batch2[key] = torch.zeros(bs2, *batch1[key].shape[1:], dtype=batch1[key].dtype)
        if key == 'input_ids':
            max_dim = max(batch1[key].shape[1], batch2[key].shape[1])
            pad1 = [max_dim - batch1[key].shape[1], 0]
            pad2 = [max_dim - batch2[key].shape[1], 0]
            batch1[key] = F.pad(batch1[key], pad1, value=padding_value)
            batch2[key] = F.pad(batch2[key], pad2, value=padding_value)
        elif key == 'attention_mask':
            max_dim = max(batch1[key].shape[1], batch2[key].shape[1])
            pad1 = [max_dim - batch1[key].shape[1], 0]
            pad2 = [max_dim - batch2[key].shape[1], 0]
            batch1[key] = F.pad(batch1[key], pad1, value=0)
            batch2[key] = F.pad(batch2[key], pad2, value=0)
        if isinstance(batch1[key], torch.Tensor):
            merged[key] = torch.cat([batch1[key], batch2[key]], dim=0)
        elif isinstance(batch1[key], list):
            merged[key] = batch1[key] + batch2[key]
        else:
            raise TypeError(f"Unsupported type for merging: {key}, {type(batch1[key])}")

    for key in batch2:
        if key not in merged:
            batch1[key] = torch.zeros(bs1, *batch2[key].shape[1:], dtype=batch2[key].dtype)
            merged[key] = torch.cat([batch1[key], batch2[key]], dim=0)
    return merged


def train_with_grpo_mm(model, processor, train_data, eval_data_deepcad, eval_data_fusion, eval_data_text, sampler, batch_size=4,
                    num_generations=4, top_samples=None, max_completion_length=128,
                    learning_rate=5e-6, batch_updates=3, epsilon_high=0.2, epsilon_low=0.2, kl_coef=0.02,
                    kl_target_low=0.1, kl_target_high=0.5, train_epochs=1,
                    reward_function=None, collate_fn=None, run_id=None, gpg=False, use_buffer=False, save_path="./models", num_workers=2,
                    temperature=1.0, do_sample=True, top_p=0.95, test_save_steps=None, config=None,
                    reward_eval_interval_steps=50, reward_eval_size=128, reward_eval_batch_size=32,
                    reward_eval_generations=1, quick_eval_interval_steps=100, quick_eval_size=200,
                    quick_eval_batch_size=64, full_eval_interval_epochs=1):

    rank = dist.get_rank()
    rank = rank % torch.cuda.device_count()

    if top_samples is None:
        top_samples = num_generations

    def is_full_eval_epoch(epoch_idx):
        if full_eval_interval_epochs is None or full_eval_interval_epochs <= 0:
            return epoch_idx == train_epochs - 1
        return ((epoch_idx + 1) % full_eval_interval_epochs == 0) or (epoch_idx == train_epochs - 1)

    def get_eval_subset(dataset, max_eval_samples):
        if dataset is None or max_eval_samples is None or max_eval_samples <= 0 or max_eval_samples >= len(dataset):
            return dataset
        return Subset(dataset, range(max_eval_samples))

    def collect_eval_metrics(dataset, dataset_name, batch_size, metric_prefix, max_eval_samples=None):
        eval_logs = {}
        if dataset is None:
            return eval_logs

        for mode in ("pc", "img"):
            dataset.mode = mode
            eval_dataset = get_eval_subset(dataset, max_eval_samples)
            ious, cds, incorrect, failed_intersect = evaluate_model_mm(
                config, model.module, processor, eval_dataset, 0, collate_fn, batch_size=batch_size
            )
            eval_logs.update({
                f"{metric_prefix}/{mode}/{dataset_name}/IoU mean": np.mean(ious),
                f"{metric_prefix}/{mode}/{dataset_name}/CD mean": np.mean(cds),
                f"{metric_prefix}/{mode}/{dataset_name}/IoU median": np.median(ious),
                f"{metric_prefix}/{mode}/{dataset_name}/CD median": np.median(cds),
                f"{metric_prefix}/{mode}/{dataset_name}/Failures fraction": incorrect,
                f"{metric_prefix}/{mode}/{dataset_name}/Intersect failure fraction": failed_intersect,
            })
        return eval_logs

    def maybe_run_reward_eval(current_step):
        if rank != 0 or config is None or reward_function is None:
            return
        if reward_eval_interval_steps is None or reward_eval_interval_steps <= 0:
            return
        if current_step <= 0 or (current_step % reward_eval_interval_steps) != 0:
            return

        reward_logs = {}
        eval_specs = [("DeepCAD", eval_data_deepcad)] if eval_data_deepcad is not None else []
        for dataset_name, dataset in eval_specs:
            for mode in ("pc", "img"):
                dataset.mode = mode
                reward_metrics = evaluate_reward_mm(
                    config,
                    model.module,
                    processor,
                    dataset,
                    0,
                    collate_fn,
                    reward_function,
                    batch_size=reward_eval_batch_size,
                    num_generations=reward_eval_generations,
                    max_eval_samples=reward_eval_size,
                )
                reward_logs.update({
                    f"reward_eval/{mode}/{dataset_name}/mean": reward_metrics["reward_mean"],
                    f"reward_eval/{mode}/{dataset_name}/median": reward_metrics["reward_median"],
                    f"reward_eval/{mode}/{dataset_name}/std": reward_metrics["reward_std"],
                    f"reward_eval/{mode}/{dataset_name}/failure_ratio": reward_metrics["failure_ratio"],
                    f"reward_eval/{mode}/{dataset_name}/num_samples": reward_metrics["num_reward_samples"],
                })

        reward_logs["step"] = current_step
        wandb.log(reward_logs)

    def maybe_run_quick_eval(current_step):
        if rank != 0 or config is None:
            return
        if quick_eval_interval_steps is None or quick_eval_interval_steps <= 0:
            return
        if current_step <= 0 or (current_step % quick_eval_interval_steps) != 0:
            return

        quick_logs = {}
        quick_logs.update(
            collect_eval_metrics(
                eval_data_deepcad,
                "DeepCAD",
                batch_size=quick_eval_batch_size,
                metric_prefix="quick_eval",
                max_eval_samples=quick_eval_size,
            )
        )
        if eval_data_fusion is not None:
            quick_logs.update(
                collect_eval_metrics(
                    eval_data_fusion,
                    "Fusion360",
                    batch_size=quick_eval_batch_size,
                    metric_prefix="quick_eval",
                    max_eval_samples=quick_eval_size,
                )
            )
        quick_logs["step"] = current_step
        wandb.log(quick_logs)

    step = 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()
    loss_fn = grpo_loss if not gpg else gpg_loss
    for epoch in range(train_epochs):
        train_data.swap()
        dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, sampler=sampler,
                                num_workers=num_workers)
        buffer = IndexBuffer()
        print(f"\nEpoch {epoch + 1}/{train_epochs}")
        # Inner loop: your original training steps.
        for batch_samples in dataloader:
            if use_buffer and len(buffer) > 0:
                indices = buffer.sample(min(batch_size, len(buffer)))
                samples = [train_data[i] for i in indices]
                buffer_batch = collate_fn(samples)
                batch_samples = merge_collated_batches(batch_samples, buffer_batch, padding_value=processor.tokenizer.pad_token_id)
            rollout_data, stats = generate_rollout_data(
                model.module,
                reward_function,
                processor,
                batch_samples,
                num_generations,
                max_completion_length,
                top_samples=top_samples,
                gpg=gpg,
                buffer=buffer,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
            )

            # Skip if all samples were filtered out
            if rollout_data is None:
                print("[SKIP] Batch skipped due to low reward variance", flush=True)
                continue

            # Dynamic KL coefficient (will be updated each iteration)
            current_kl_coef = kl_coef

            for grpo_iter in range(batch_updates):
                # First pass: compute loss to get current KL
                loss, entropy, kl, approx_kl, ratio_mean = loss_fn(
                    model.module,
                    rollout_data,
                    processor,
                    reward_function,
                    epsilon_high=epsilon_high,
                    epsilon_low=epsilon_low,
                    top_samples=top_samples,
                    kl_coef=current_kl_coef
                )

                # Skip abnormal steps
                if approx_kl > 10 or ratio_mean > 10:
                    if rank == 0:
                        print(f"WARNING: Skipping step due to abnormal values - approx_kl: {approx_kl:.4f}, ratio_mean: {ratio_mean:.4f}")
                    continue

                # Update KL coefficient for next iteration based on current KL
                if kl > kl_target_high:
                    current_kl_coef = 0.05
                elif kl < kl_target_low:
                    current_kl_coef = 0.01
                else:
                    current_kl_coef = kl_coef

                optimizer.zero_grad()
                loss.backward()
                torch.cuda.synchronize()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                optimizer.step()
                torch.cuda.synchronize()
                # Log to wandb
                if rank == 0:
                    wandb.log({
                        "loss": loss.item(),
                        "average_reward": stats["avg_reward"],
                        "reward_std": stats["reward_std"],
                        "invalid_ratio": stats["invalid_ratio"],
                        "unique_completions": stats["unique_completions"],
                        "total_completions": stats["total_completions"],
                        "entropy": entropy,
                        "kl_divergence": kl,
                        "approx_kl": approx_kl,
                        "ratio_mean": ratio_mean,
                        "grad_norm": grad_norm.item(),
                        "kl_coef": current_kl_coef,
                        "step": step + 1,
                        "grpo_iter": grpo_iter + 1,
                        "iter epoch": epoch + 1,
                    })
                    print(f"Epoch {epoch + 1}/{train_epochs}, Step {step + 1}/{len(dataloader)}, "
                          f"GRPO iter {grpo_iter + 1}/{batch_updates}, loss: {loss.item():.4f}, "
                          f"reward: {stats['avg_reward']:.4f}±{stats['reward_std']:.4f}, "
                          f"invalid: {stats['invalid_ratio']:.2%}, unique: {stats['unique_completions']}/{stats['total_completions']}, "
                          f"entropy: {entropy:.4f}, kl: {kl:.4f}, β: {current_kl_coef:.3f}, ratio: {ratio_mean:.4f}, grad: {grad_norm.item():.4f}")
            step += 1

            # Test save: exit after N steps
            if test_save_steps is not None and step >= test_save_steps:
                if rank == 0:
                    print(f"\n[TEST SAVE] Reached {test_save_steps} steps, saving model and exiting...")
                    model.module.save_pretrained(f"{save_path}/test_save_{run_id}")
                    processor.save_pretrained(f"{save_path}/test_save_{run_id}")
                    print(f"[TEST SAVE] Model saved to {save_path}/test_save_{run_id}")
                dist.barrier()
                return model.module

            maybe_run_reward_eval(step)
            maybe_run_quick_eval(step)

        if rank == 0 and is_full_eval_epoch(epoch):
            if config is None:
                raise ValueError("train_with_grpo_mm requires config for evaluation and wandb logging")

            eval_logs = {}
            eval_logs.update(
                collect_eval_metrics(
                    eval_data_deepcad,
                    "DeepCAD test",
                    batch_size=200,
                    metric_prefix="eval",
                )
            )
            if eval_data_fusion is not None:
                eval_logs.update(
                    collect_eval_metrics(
                        eval_data_fusion,
                        "Fusion360 test",
                        batch_size=200,
                        metric_prefix="eval",
                    )
                )

            model.module.save_pretrained(f"{save_path}/{run_id}_{epoch}")
            processor.save_pretrained(f"{save_path}/{run_id}_{epoch}")
            eval_logs["epoch"] = epoch + 1
            wandb.log(eval_logs)


        # if rank == 0:
        #     eval_data_deepcad.mode = 'pc'
        #     ious, cds, incorrect, failed_intersect = evaluate_model_mm(
        #         model.module,
        #         processor,
        #         eval_data_deepcad,
        #         0,
        #         collate_fn,
        #         batch_size=200,
        #     )

        #     eval_data_deepcad.mode = 'img'
        #     ious_im, cds_im, incorrect_im, failed_intersect_im = evaluate_model_mm(
        #         model.module,
        #         processor,
        #         eval_data_deepcad,
        #         0,
        #         collate_fn,
        #         batch_size=200,
        #     )

        #     model.module.save_pretrained(f"{save_path}/{run_id}_{epoch}")
        #     processor.save_pretrained(f"{save_path}/{run_id}_{epoch}")

        #     wandb.log({
        #         "eval/pc/DeepCAD test/IoU mean": np.mean(ious),
        #         "eval/pc/DeepCAD test/CD mean": np.mean(cds),
        #         "eval/pc/DeepCAD test/IoU median": np.median(ious),
        #         "eval/pc/DeepCAD test/CD median": np.median(cds),
        #         "eval/pc/DeepCAD test/Failures fraction": incorrect,
        #         "eval/img/DeepCAD test/IoU mean": np.mean(ious_im),
        #         "eval/img/DeepCAD test/CD mean": np.mean(cds_im),
        #         "eval/img/DeepCAD test/IoU median": np.median(ious_im),
        #         "eval/img/DeepCAD test/CD median": np.median(cds_im),
        #         "eval/img/DeepCAD test/Failures fraction": incorrect_im,
        #     })

        dist.barrier()
    return model.module
