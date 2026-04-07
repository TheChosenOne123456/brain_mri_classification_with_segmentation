'''
case index 的加载 / 保存
和文件系统打交道，但不关心 MRI 语义
'''

import json
from pathlib import Path

INDEX_FILE_NAME = "case_index.json"
# 定义单独存放mask索引的文件名
MASK_INDEX_FILE_NAME = "mask_index.json"

def load_index(index_path: Path):
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_index(index, index_path: Path):
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
