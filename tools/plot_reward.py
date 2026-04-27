#!/usr/bin/env python
"""Plot reward curve from exported wandb metrics."""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_file", help="CSV file with metrics")
    parser.add_argument("--output", default=None, help="output image file")
    parser.add_argument("--smooth", type=float, default=0.9, help="smoothing factor (0-1)")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)

    # Find step column
    step_col = 'step' if 'step' in df.columns else '_step'

    # Drop rows with NaN in step or reward columns
    df = df.dropna(subset=[step_col])

    # Find reward column
    reward_col = None
    for col in ['average_reward', 'reward', 'avg_reward']:
        if col in df.columns:
            reward_col = col
            break

    if reward_col is None:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError("No reward column found")

    df = df.dropna(subset=[reward_col])

    # Exponential moving average smoothing
    rewards = df[reward_col].values
    smoothed = np.zeros_like(rewards)
    smoothed[0] = rewards[0]
    for i in range(1, len(rewards)):
        smoothed[i] = args.smooth * smoothed[i-1] + (1 - args.smooth) * rewards[i]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df[step_col], rewards, alpha=0.3, linewidth=1, label='Raw')
    plt.plot(df[step_col], smoothed, linewidth=2, label='Smoothed')
    plt.xlabel('Step')
    plt.ylabel('Average Reward')
    plt.title('Training Reward Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output = args.output or args.csv_file.replace('.csv', '.png')
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output}")

if __name__ == "__main__":
    main()
