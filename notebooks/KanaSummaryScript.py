import os
import re
import ast
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import statistics

ROOT = "/root/autodl-tmp/Semi-supervised-learning/notebooks/saved_results/kana_experiment"
OUT  = "/root/autodl-tmp/Semi-supervised-learning/notebooks/saved_results/kana_experiment/_summary"

os.makedirs(OUT, exist_ok=True)

NET_CANDIDATES = ["resnet50", "vit_tiny_patch2_32"]
DATASET_CANDIDATES = ["kana"]

LOSS_RE = re.compile(r'^Iter\s+(\d+):\s+(.*)$')
METRIC_RE = re.compile(r'^(acc|precision|recall|f1)\s*:\s*([0-9.]+)')

def parse_name(name):
    m = re.search(
        r'_labels(?P<labels>[\d\.]+)_train(?P<train>\d+)_val(?P<val>\d+)_test(?P<test>\d+)_classes(?P<classes>\d+)_epoch(?P<epoch>\d+)_lr(?P<lr>[\d\.eE-]+)_seed(?P<seed>\d+)$',
        name
    )
    if not m:
        return None

    tail = m.groupdict()
    prefix = name[:m.start()]  # algorithm_dataset_net

    net = None
    for cand in NET_CANDIDATES:
        if prefix.endswith("_" + cand):
            net = cand
            prefix = prefix[:-(len(cand)+1)]
            break

    dataset = None
    for cand in DATASET_CANDIDATES:
        if prefix.endswith("_" + cand):
            dataset = cand
            algorithm = prefix[:-(len(cand)+1)]
            break
    else:
        if "_" in prefix:
            algorithm, dataset = prefix.rsplit("_", 1)
        else:
            algorithm, dataset = prefix, ""

    return {
        "save_name": name,
        "algorithm": algorithm,
        "dataset": dataset,
        "net": net if net else "",
        "num_labels": float(tail["labels"]),
        "num_train_iter": int(tail["train"]),
        "num_eval_iter": int(tail["val"]),
        "num_test_iter": int(tail["test"]),
        "num_classes": int(tail["classes"]),
        "epoch": int(tail["epoch"]),
        "lr": float(tail["lr"]),
        "seed": int(tail["seed"])
    }

def parse_train_log(path):
    loss_records = []
    eval_records = []
    pending_conf = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    in_eval = False
    current_eval = {}
    conf_collect = False
    conf_buf = []

    def finalize_eval():
        nonlocal current_eval, pending_conf, eval_records, in_eval
        if current_eval:
            if pending_conf is not None and "conf_mat" not in current_eval:
                current_eval["conf_mat"] = pending_conf
                pending_conf = None
            eval_records.append(current_eval)
        current_eval = {}
        in_eval = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = LOSS_RE.match(line)
        if m:
            it = int(m.group(1))
            rest = m.group(2)
            parts = [p.strip() for p in rest.split("|")]
            rec = {"iter": it}
            for p in parts:
                if ":" in p:
                    k, v = p.split(":", 1)
                    try:
                        rec[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            loss_records.append(rec)
            continue

        if "confusion matrix" in line:
            conf_collect = True
            conf_buf = []
            continue

        if conf_collect:
            conf_buf.append(line)
            if "]]" in line:
                conf_collect = False
                try:
                    s = " ".join(conf_buf)
                    mat = np.array(ast.literal_eval(s))
                    pending_conf = mat
                except Exception:
                    pending_conf = None
            continue

        if "evaluation metric" in line:
            if in_eval:
                finalize_eval()
            in_eval = True
            current_eval = {"eval_idx": len(eval_records)}
            continue

        if in_eval:
            mm = METRIC_RE.match(line)
            if mm:
                k, v = mm.group(1), mm.group(2)
                current_eval[k] = float(v)
                continue
            else:
                finalize_eval()
                continue

    if in_eval:
        finalize_eval()

    return loss_records, eval_records

def topk_avg_metrics(metrics, k=10):
    if not metrics:
        return None
    ranked = sorted(metrics, key=lambda m: m.get("acc", -1), reverse=True)
    top = ranked[:k]
    total = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    for m in top:
        for kk in total:
            total[kk] += m.get(kk, 0.0)
    count = float(len(top))
    return {kk: total[kk] / count for kk in total}

def plot_loss_curve(loss_records, out_path, keys):
    if not loss_records:
        return
    iters = [r["iter"] for r in loss_records if "iter" in r]
    if not iters:
        return
    plt.figure(figsize=(8,4))
    for k in keys:
        ys = [r.get(k, None) for r in loss_records]
        if any(v is not None for v in ys):
            plt.plot(iters, ys, label=k)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_eval_curve(eval_records, out_path):
    if not eval_records:
        return
    xs = [r["eval_idx"] for r in eval_records]
    plt.figure(figsize=(8,4))
    for k in ["acc", "precision", "recall", "f1"]:
        ys = [r.get(k, None) for r in eval_records]
        if any(v is not None for v in ys):
            plt.plot(xs, ys, label=k)
    plt.xlabel("epoch index")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_best_final_bar(best_acc, final_acc, out_path):
    plt.figure(figsize=(4,4))
    plt.bar(["best", "final"], [best_acc, final_acc], color=["tab:green", "tab:blue"])
    plt.ylabel("acc")
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_confmat_heatmap(mat, out_path, title="confusion matrix"):
    if mat is None:
        return
    plt.figure(figsize=(4,4))
    plt.imshow(mat, cmap="Blues", vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar()
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_loss_vs_acc(loss_records, eval_records, iters_per_epoch, out_path):
    if not loss_records or not eval_records or iters_per_epoch is None:
        return
    iters = [r["iter"] for r in loss_records]
    total_loss = [r.get("train/total_loss", None) for r in loss_records]
    eval_x = [(r["eval_idx"] + 1) * iters_per_epoch for r in eval_records]
    eval_acc = [r.get("acc", None) for r in eval_records]

    if not any(v is not None for v in total_loss):
        return

    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(iters, total_loss, color="tab:red", label="train/total_loss")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("loss", color="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(eval_x, eval_acc, color="tab:blue", marker="o", label="eval acc")
    ax2.set_ylabel("acc", color="tab:blue")
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close()

def mean_std(values):
    if not values:
        return "", ""
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)

# 收集实验
experiments = []
for entry in os.scandir(ROOT):
    if not entry.is_dir():
        continue
    train_log = os.path.join(entry.path, "train.log")
    if not os.path.exists(train_log):
        continue
    meta = parse_name(entry.name)
    if meta is None:
        continue
    loss_records, eval_records = parse_train_log(train_log)
    meta["loss_records"] = loss_records
    meta["eval_records"] = eval_records
    experiments.append(meta)

print(f"Found {len(experiments)} experiments")

# summary.csv
summary_path = os.path.join(OUT, "summary.csv")
with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "save_name","algorithm","dataset","net","num_labels","seed",
        "best_acc","best_precision","best_recall","best_f1",
        "final_acc","final_precision","final_recall","final_f1",
        "top10_avg_acc","top10_avg_precision","top10_avg_recall","top10_avg_f1",
        "best_eval_idx","final_eval_idx"
    ])
    for exp in experiments:
        evals = exp["eval_records"]
        if not evals:
            continue
        best = max(evals, key=lambda r: r.get("acc", -1))
        final = evals[-1]
        top10 = topk_avg_metrics(evals, k=10)
        writer.writerow([
            exp["save_name"], exp["algorithm"], exp["dataset"], exp["net"],
            exp["num_labels"], exp["seed"],
            best.get("acc", None), best.get("precision", None), best.get("recall", None), best.get("f1", None),
            final.get("acc", None), final.get("precision", None), final.get("recall", None), final.get("f1", None),
            top10.get("acc", None) if top10 else None,
            top10.get("precision", None) if top10 else None,
            top10.get("recall", None) if top10 else None,
            top10.get("f1", None) if top10 else None,
            best.get("eval_idx", None), final.get("eval_idx", None)
        ])

# 每个实验输出图
EXP_DIR = os.path.join(OUT, "experiments")
os.makedirs(EXP_DIR, exist_ok=True)

for exp in experiments:
    exp_dir = os.path.join(EXP_DIR, exp["save_name"])
    os.makedirs(exp_dir, exist_ok=True)

    plot_loss_curve(exp["loss_records"], os.path.join(exp_dir, "loss_curve.png"),
                    keys=["train/total_loss","train/sup_loss","train/unsup_loss"])

    plot_loss_curve(exp["loss_records"], os.path.join(exp_dir, "util_ratio.png"),
                    keys=["train/util_ratio"])

    plot_eval_curve(exp["eval_records"], os.path.join(exp_dir, "eval_curve.png"))

    if exp["eval_records"]:
        best = max(exp["eval_records"], key=lambda r: r.get("acc", -1))
        final = exp["eval_records"][-1]
        plot_best_final_bar(best.get("acc", 0), final.get("acc", 0),
                            os.path.join(exp_dir, "best_vs_final.png"))

        if best.get("conf_mat", None) is not None:
            plot_confmat_heatmap(best["conf_mat"], os.path.join(exp_dir, "confmat_best.png"), "confmat (best)")
        if final.get("conf_mat", None) is not None:
            plot_confmat_heatmap(final["conf_mat"], os.path.join(exp_dir, "confmat_final.png"), "confmat (final)")

    iters_per_epoch = None
    if exp["epoch"] and exp["num_train_iter"]:
        iters_per_epoch = exp["num_train_iter"] / exp["epoch"]
    plot_loss_vs_acc(exp["loss_records"], exp["eval_records"], iters_per_epoch,
                     os.path.join(exp_dir, "loss_vs_acc.png"))

# 对比图
labels_plot_path = os.path.join(OUT, "acc_vs_labels_by_net.png")
plt.figure(figsize=(7,4))
for net in sorted(set(e["net"] for e in experiments if e["net"])):
    xs = []
    ys = []
    es = []
    for num_labels in sorted(set(e["num_labels"] for e in experiments)):
        vals = []
        for e in experiments:
            if e["net"] == net and e["num_labels"] == num_labels and e["eval_records"]:
                best = max(e["eval_records"], key=lambda r: r.get("acc", -1))
                vals.append(best.get("acc", None))
        vals = [v for v in vals if v is not None]
        if vals:
            xs.append(num_labels)
            ys.append(np.mean(vals))
            es.append(np.std(vals))
    if xs:
        plt.errorbar(xs, ys, yerr=es, marker="o", label=net)
plt.xlabel("num_labels")
plt.ylabel("best acc (mean±std)")
plt.legend()
plt.tight_layout()
plt.savefig(labels_plot_path)
plt.close()

net_box_path = os.path.join(OUT, "acc_by_net_box.png")
net_vals = {}
for e in experiments:
    if not e["eval_records"]:
        continue
    best = max(e["eval_records"], key=lambda r: r.get("acc", -1))
    net_vals.setdefault(e["net"], []).append(best.get("acc", None))
labels = [k for k in net_vals.keys() if k]
data = [[v for v in net_vals[k] if v is not None] for k in labels]
if data:
    plt.figure(figsize=(6,4))
    plt.boxplot(data, tick_labels=labels, showmeans=True)
    plt.ylabel("best acc")
    plt.tight_layout()
    plt.savefig(net_box_path)
    plt.close()

seed_plot_path = os.path.join(OUT, "acc_by_seed_mean_std.png")
seed_vals = {}
for e in experiments:
    if not e["eval_records"]:
        continue
    best = max(e["eval_records"], key=lambda r: r.get("acc", -1))
    seed_vals.setdefault(e["seed"], []).append(best.get("acc", None))
seeds = sorted(seed_vals.keys())
means = [np.mean([v for v in seed_vals[s] if v is not None]) for s in seeds]
stds  = [np.std([v for v in seed_vals[s] if v is not None]) for s in seeds]
if seeds:
    plt.figure(figsize=(8,4))
    plt.errorbar(seeds, means, yerr=stds, fmt="o")
    plt.xlabel("seed")
    plt.ylabel("best acc (mean±std)")
    plt.tight_layout()
    plt.savefig(seed_plot_path)
    plt.close()

# 分组统计（按命名规则字段，不受 algorithm 下划线影响）
group_rows = []
groups = {}
for exp in experiments:
    group_key = (
        exp["algorithm"], exp["dataset"], exp["net"],
        exp["num_labels"], exp["num_train_iter"], exp["num_eval_iter"],
        exp["num_classes"], exp["epoch"], exp["lr"]
    )
    g = groups.setdefault(group_key, {"seeds": [], "rows": []})
    g["seeds"].append(exp["seed"])
    g["rows"].append(exp)

metric_keys = [
    "best_acc","best_precision","best_recall","best_f1",
    "final_acc","final_precision","final_recall","final_f1",
    "top10_avg_acc","top10_avg_precision","top10_avg_recall","top10_avg_f1"
]

for group_key, g in groups.items():
    algorithm, dataset, net, num_labels, num_train_iter, num_eval_iter, num_classes, epoch, lr = group_key
    out = {
        "algorithm": algorithm,
        "dataset": dataset,
        "net": net,
        "num_labels": num_labels,
        "num_train_iter": num_train_iter,
        "num_eval_iter": num_eval_iter,
        "num_classes": num_classes,
        "epoch": epoch,
        "lr": lr,
        "n": len(g["rows"]),
        "seeds": ",".join(str(s) for s in sorted(set(g["seeds"]))),
    }

    for k in metric_keys:
        vals = []
        for exp in g["rows"]:
            evals = exp["eval_records"]
            if not evals:
                continue
            best = max(evals, key=lambda r: r.get("acc", -1))
            final = evals[-1]
            top10 = topk_avg_metrics(evals, k=10)

            if k == "best_acc":
                v = best.get("acc", None)
            elif k == "best_precision":
                v = best.get("precision", None)
            elif k == "best_recall":
                v = best.get("recall", None)
            elif k == "best_f1":
                v = best.get("f1", None)
            elif k == "final_acc":
                v = final.get("acc", None)
            elif k == "final_precision":
                v = final.get("precision", None)
            elif k == "final_recall":
                v = final.get("recall", None)
            elif k == "final_f1":
                v = final.get("f1", None)
            elif k == "top10_avg_acc":
                v = top10.get("acc", None) if top10 else None
            elif k == "top10_avg_precision":
                v = top10.get("precision", None) if top10 else None
            elif k == "top10_avg_recall":
                v = top10.get("recall", None) if top10 else None
            elif k == "top10_avg_f1":
                v = top10.get("f1", None) if top10 else None
            else:
                v = None

            if isinstance(v, (int, float)):
                vals.append(v)

        m, s = mean_std(vals)
        out[f"{k}_mean"] = m
        out[f"{k}_std"] = s

    group_rows.append(out)

group_rows.sort(key=lambda r: (r["algorithm"], r["dataset"], r["net"], r["num_labels"], r["seed"] if "seed" in r else 0))

group_output = os.path.join(OUT, "summary_by_seed.csv")
group_fieldnames = [
    "algorithm","dataset","net","num_labels","num_train_iter","num_eval_iter",
    "num_classes","epoch","lr","n","seeds"
]
for k in metric_keys:
    group_fieldnames.append(f"{k}_mean")
    group_fieldnames.append(f"{k}_std")

with open(group_output, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=group_fieldnames)
    writer.writeheader()
    writer.writerows(group_rows)

print("Done. Outputs in:", OUT)

