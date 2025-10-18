import json
import argparse
import os
from typing import Dict, List, Set, Any


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def to_set(xs: Any) -> Set[str]:
    if xs is None:
        return set()
    if isinstance(xs, list):
        return set(map(str, xs))
    # Fallback for bad types
    try:
        return set(map(str, xs))
    except Exception:
        return set()


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return 2.0 * tp / denom


def main():
    parser = argparse.ArgumentParser(description='评估精确率(Precision)、召回率(Recall)、F1 分数')
    parser.add_argument('--val_path', default='datasets/CDrugRed-A-v1/CDrugRed_val.jsonl', type=str,
                        help='验证集 JSONL（包含就诊标识与出院带药列表）')
    parser.add_argument('--pred_path', default='output/CDrugRed_val_pred.json', type=str,
                        help='预测结果 JSON（格式为[{"ID":..., "prediction": [...]}, ...]）')
    parser.add_argument('--save_report', default='output/CDrugRed_val_metrics.json', type=str,
                        help='评估指标保存路径（JSON）')
    parser.add_argument('--macro', choices=['nonempty', 'all'], default='nonempty',
                        help='宏平均范围：nonempty 仅统计有真实药单的样本；all 统计全部样本')
    parser.add_argument('--per_sample_out', default=None, type=str,
                        help='可选：输出每条样本的 P/R/F1 明细 JSONL 路径')

    args = parser.parse_args()

    # 读取验证集
    val_items = read_jsonl(args.val_path)
    id_to_gt: Dict[str, Set[str]] = {}
    for rec in val_items:
        pid = rec.get('就诊标识')
        gt = to_set(rec.get('出院带药列表'))
        if pid is not None:
            id_to_gt[str(pid)] = gt

    # 读取预测文件
    with open(args.pred_path, 'r', encoding='utf-8') as f:
        preds_list = json.load(f)

    id_to_pred: Dict[str, Set[str]] = {}
    for it in preds_list:
        pid = it.get('ID')
        pred = to_set(it.get('prediction', []))
        if pid is not None:
            id_to_pred[str(pid)] = pred

    # 统计缺失/多余预测
    missing_pred_ids = [pid for pid in id_to_gt.keys() if pid not in id_to_pred]
    extra_pred_ids = [pid for pid in id_to_pred.keys() if pid not in id_to_gt]

    # 逐样本评估
    per_sample = []
    micro_tp = micro_fp = micro_fn = 0
    macro_pool = []  # 用于宏平均（precision/recall/f1）
    jaccard_pool = []  # 用于样本级Jaccard的平均（与macro范围一致）

    for pid, gt in id_to_gt.items():
        pred = id_to_pred.get(pid, set())
        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)

        # per-sample precision/recall/f1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = f1_from_counts(tp, fp, fn)

        # per-sample Jaccard: |gt ∩ pred| / |gt ∪ pred|，并集为空时设为 1.0
        union_size = len(gt | pred)
        inter_size = len(gt & pred)
        jacc = 1.0 if union_size == 0 else (inter_size / union_size)

        per_sample.append({
            'ID': pid,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': p,
            'recall': r,
            'f1': f1,
            'jaccard': jacc,
            'gt_count': len(gt),
            'pred_count': len(pred),
        })

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn

        if args.macro == 'all' or len(gt) > 0:
            macro_pool.append((p, r, f1))
            jaccard_pool.append(jacc)

    # micro metrics
    micro_p = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) > 0 else 0.0
    micro_r = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) > 0 else 0.0
    micro_f1 = f1_from_counts(micro_tp, micro_fp, micro_fn)

    # macro metrics
    if macro_pool:
        macro_p = sum(x[0] for x in macro_pool) / len(macro_pool)
        macro_r = sum(x[1] for x in macro_pool) / len(macro_pool)
        macro_f1 = sum(x[2] for x in macro_pool) / len(macro_pool)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    # average Jaccard over the same sample set used for macro (nonempty or all)
    if jaccard_pool:
        avg_jaccard = sum(jaccard_pool) / len(jaccard_pool)
    else:
        avg_jaccard = 0.0

    report = {
        'n_samples': len(id_to_gt),
        'missing_pred': len(missing_pred_ids),
        'extra_pred': len(extra_pred_ids),
        'micro': {
            'tp': micro_tp,
            'fp': micro_fp,
            'fn': micro_fn,
            'precision': micro_p,
            'recall': micro_r,
            'f1': micro_f1,
        },
        'macro': {
            'avg_over': 'nonempty_gt' if args.macro == 'nonempty' else 'all_samples',
            'precision': macro_p,
            'recall': macro_r,
            'f1': macro_f1,
            'count': len(macro_pool),
        },
        'jaccard': avg_jaccard,
        'score': (avg_jaccard + macro_f1) / 2.0,
    }

    # 输出控制台
    print('Evaluation summary:')
    print(f"Samples: {report['n_samples']} | Missing pred: {report['missing_pred']} | Extra pred: {report['extra_pred']}")
    print(f"Micro  P/R/F1: {micro_p:.4f} / {micro_r:.4f} / {micro_f1:.4f}")
    print(f"Macro({report['macro']['avg_over']}) P/R/F1: {macro_p:.4f} / {macro_r:.4f} / {macro_f1:.4f}")
    print(f"Jaccard(avg over {report['macro']['avg_over']}): {avg_jaccard:.4f}")
    print(f"Score = (Jaccard + Macro F1)/2: {report['score']:.4f}")

    # 保存报表
    os.makedirs(os.path.dirname(args.save_report), exist_ok=True)
    with open(args.save_report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to: {args.save_report}")

    # 可选输出每条样本明细
    if args.per_sample_out:
        os.makedirs(os.path.dirname(args.per_sample_out), exist_ok=True)
        with open(args.per_sample_out, 'w', encoding='utf-8') as f:
            for row in per_sample:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Per-sample metrics written to: {args.per_sample_out}")


if __name__ == '__main__':
    main()