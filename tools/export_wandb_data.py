#!/usr/bin/env env python
"""Export training metrics from wandb to CSV for plotting."""
import argparse
import wandb
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default=None, help="wandb entity (username)")
    parser.add_argument("--project", default="CAD_test", help="wandb project name")
    parser.add_argument("--run_id", required=True, help="wandb run ID")
    parser.add_argument("--output", default="metrics.csv", help="output CSV file")
    args = parser.parse_args()

    api = wandb.Api()

    # Try with entity if provided, otherwise try without
    if args.entity:
        path = f"{args.entity}/{args.project}/{args.run_id}"
    else:
        path = f"{args.project}/{args.run_id}"

    try:
        run = api.run(path)
    except:
        # If failed, try to get entity from API
        print(f"Failed with path: {path}")
        print("Trying to fetch entity from API...")
        runs = api.runs(args.project)
        matching = [r for r in runs if r.id == args.run_id]
        if matching:
            run = matching[0]
            print(f"Found run: {run.entity}/{run.project}/{run.id}")
        else:
            raise ValueError(f"Could not find run with ID: {args.run_id}")

    # 获取所有历史数据（不限制行数）
    history = run.history(samples=1000000)
    history.to_csv(args.output, index=False)
    print(f"Exported {len(history)} rows to {args.output}")
    print(f"Columns: {list(history.columns)}")

if __name__ == "__main__":
    main()
