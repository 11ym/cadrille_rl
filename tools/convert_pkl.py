#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
from pprint import pprint


def convert_rich_to_light(input_pkl, output_pkl):
    print(f"准备读取: {input_pkl}")
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"输入文件不存在: {input_pkl}")

    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise TypeError(f"pkl内容不是 list，而是: {type(data)}")

    print(f"读取成功，样本数: {len(data)}")

    new_data = []
    skipped = 0

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            print(f"[警告] 第 {i} 个样本不是 dict，跳过，类型: {type(item)}")
            skipped += 1
            continue

        if "dataset" not in item or "file_name" not in item or "gt_mesh_path" not in item:
            print(f"[警告] 第 {i} 个样本缺少必要字段，跳过。keys={list(item.keys())}")
            skipped += 1
            continue

        new_item = {
            "dataset": item["dataset"],
            "file_name": item["file_name"],
            "gt_mesh_path": item["gt_mesh_path"],
        }
        new_data.append(new_item)

    print(f"转换完成，有效样本数: {len(new_data)}，跳过: {skipped}")

    with open(output_pkl, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"已保存到: {output_pkl}")

    if len(new_data) > 0:
        print("\n新文件第 0 个样本预览:")
        pprint(new_data[0])


def main():
    if len(sys.argv) < 3:
        print("用法:")
        print("python convert_rich_pkl.py 输入.pkl 输出.pkl")
        sys.exit(1)

    input_pkl = sys.argv[1]
    output_pkl = sys.argv[2]

    convert_rich_to_light(input_pkl, output_pkl)


if __name__ == "__main__":
    main()