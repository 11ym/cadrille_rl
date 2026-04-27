# CAD Code Generation with RL Fine-tuning

This repository implements an extension of a multimodal CAD code generation framework with online reinforcement learning fine-tuning.

The project focuses on generating executable Python CAD code from point cloud, image, and text inputs. Based on a supervised fine-tuning backbone, an online RL module is introduced to improve the geometric quality and executability of generated CAD programs.

## Overview

- Support multimodal inputs: point cloud, image, and text
- Generate executable Python CAD code based on CadQuery
- Extend the SFT baseline with online RL fine-tuning
- Implement GRPO/CPPO-style policy optimization
- Design geometry-based rewards using IoU and Chamfer Distance
- Apply hard sample mining for more efficient training
- Evaluate generated programs with IoU, Chamfer Distance, and Invalid Rate

## Repository Structure

```text
.
├── cadrille.py                  # Model definition: Qwen2-VL-2B with point cloud encoder
├── dataset.py                   # Dataset loading and preprocessing
├── train.py                     # Supervised fine-tuning script
├── test.py                      # Inference and CAD code generation
├── evaluate.py                  # Evaluation with IoU, CD, and IR
├── rl_finetune_notfull_222/     # Online RL fine-tuning module
├── tools/                       # Utility scripts
└── viz/                         # Visualization results
