import os
import json
import argparse
import random


def read_jsonl(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path, items):
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='将训练数据划分为 train/val，并保存到数据集目录的子目录中')
    parser.add_argument('--input_path', default='datasets/CDrugRed-A-v1/CDrugRed_train.jsonl', type=str)
    parser.add_argument('--output_subdir', default='split', type=str, help='在数据集目录下创建的子目录名，如 split 或 splits/seed42_r0.1')
    parser.add_argument('--ratio', default=0.1, type=float, help='验证集比例，0~1之间，默认0.1')
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    data = read_jsonl(args.input_path)
    n = len(data)
    if n == 0:
        raise ValueError('训练集为空')

    # Shuffle indices deterministically
    rnd = random.Random(args.seed)
    indices = list(range(n))
    rnd.shuffle(indices)

    # Compute split sizes
    if args.ratio <= 0:
        k = 0
    elif args.ratio >= 1:
        k = n
    else:
        k = int(n * args.ratio)

    val_idx = set(indices[:k])

    val_items = []
    train_items = []
    for i, rec in enumerate(data):
        if i in val_idx:
            # 保留原始的“出院带药列表”，不在划分阶段清空
            val_items.append(rec)
        else:
            train_items.append(rec)

    # Determine dataset base directory and output subdir
    dataset_dir = os.path.dirname(args.input_path)
    out_dir = os.path.join(dataset_dir, args.output_subdir)
    os.makedirs(out_dir, exist_ok=True)

    # Use consistent filenames to方便后续脚本使用
    out_train = os.path.join(out_dir, 'CDrugRed_train.jsonl')
    out_val = os.path.join(out_dir, 'CDrugRed_val.jsonl')

    write_jsonl(out_train, train_items)
    write_jsonl(out_val, val_items)

    print(f'Total: {n} | Val: {len(val_items)} | Train: {len(train_items)}')
    print(f'Train saved to: {out_train}')
    print(f'Validation saved to: {out_val}')


if __name__ == '__main__':
    main()