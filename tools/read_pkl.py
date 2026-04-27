#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pickle
from pprint import pprint


def main():
    print("脚本已启动")
    print("sys.argv =", sys.argv)

    if len(sys.argv) < 2:
        print("用法: python read_pkl.py <pkl_path>")
        sys.exit(1)

    pkl_path = sys.argv[1]
    print("准备读取:", pkl_path)

    if not os.path.exists(pkl_path):
        print("文件不存在:", pkl_path)
        sys.exit(1)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    print("读取成功")
    print("对象类型:", type(data))

    if hasattr(data, "__len__"):
        try:
            print("长度:", len(data))
        except Exception as e:
            print("无法获取长度:", e)

    if isinstance(data, list) and len(data) > 0:
        print("\n第 0 个元素类型:", type(data[0]))
        print("第 0 个元素内容:")
        pprint(data[0])

        if isinstance(data[0], dict):
            print("\nkeys:")
            print(list(data[0].keys()))

    elif isinstance(data, dict):
        print("\n前 10 个 key:")
        print(list(data.keys())[:10])

    else:
        print("\n对象预览:")
        pprint(data)


if __name__ == "__main__":
    main()