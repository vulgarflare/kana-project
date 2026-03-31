import os
import re
import glob
import numpy as np
from PIL import Image

from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ssl_data


# 文件名示例：
# 980000.jpg_cropped_2.jpg  -> label = 2
LABEL_RE = re.compile(r"_([0-9]+)\.jpg$", re.IGNORECASE)


def parse_label(fname: str) -> int:
    """从文件名中解析标签"""
    m = LABEL_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse label from filename: {fname}")
    return int(m.group(1))


def load_images_as_numpy(paths, img_size: int):
    """
    读取图像 -> resize -> RGB -> numpy uint8

    返回:
        np.ndarray [N, H, W, C], dtype=uint8
    """
    data = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size), Image.BILINEAR)
        data.append(np.asarray(img, dtype=np.uint8))
    return np.stack(data, axis=0)


def _stratified_split_eval_train(
    all_paths,
    all_targets,
    num_classes: int,
    rng: np.random.RandomState,
    val_per_class: int | None = None,
    val_ratio: float | None = None,
    min_train_per_class: int = 1,
):
    """
    按类别分层拆分 eval/train，但不再严格要求每类固定 eval 数量。
    策略：
      - 若提供 val_per_class：每类尽量取 val_per_class，但至少给 train 留 min_train_per_class（当可能时）
      - 若未提供 val_per_class：使用 val_ratio 做整体比例分层拆分
      - 极小类：
          n=1 -> 放 train（eval 0）
          n>=2 -> eval 最多 n-min_train_per_class
    返回：
      train_idx, val_idx (np.ndarray)
    """

    all_targets = np.asarray(all_targets, dtype=np.int64)
    n_total = len(all_targets)
    assert n_total == len(all_paths)

    # 收集每类索引
    per_cls = []
    for c in range(num_classes):
        idx = np.where(all_targets == c)[0]
        rng.shuffle(idx)
        per_cls.append(idx)

    val_indices = []
    train_indices = []

    if val_per_class is not None:
        # 分层：每类尽量抽 val_per_class，但确保 train 还有样本
        for c in range(num_classes):
            cls_idx = per_cls[c]
            n = len(cls_idx)

            if n <= 0:
                # 允许空类：不报错，直接跳过（但你的 split_ssl_data 可能对空类有假设）
                continue

            if n == 1:
                # 只有 1 张：放训练，eval 不放
                train_indices.extend(cls_idx.tolist())
                continue

            # 至少给 train 留 min_train_per_class（一般为 1）
            max_val = max(0, n - min_train_per_class)
            k = min(val_per_class, max_val)

            # 如果 k==0，说明 n==min_train_per_class（例如 n=1 已处理；n=1/0外，这里可能 n=1）
            if k > 0:
                val_indices.extend(cls_idx[:k].tolist())
                train_indices.extend(cls_idx[k:].tolist())
            else:
                train_indices.extend(cls_idx.tolist())

    else:
        # 使用 val_ratio 做整体分层采样（默认会给 train 留 1）
        if val_ratio is None:
            val_ratio = 0.1  # 合理默认值

        val_ratio = float(val_ratio)
        val_ratio = max(0.0, min(0.9, val_ratio))

        for c in range(num_classes):
            cls_idx = per_cls[c]
            n = len(cls_idx)

            if n <= 0:
                continue

            if n == 1:
                train_indices.extend(cls_idx.tolist())
                continue

            # 目标 eval 数 = round(n*val_ratio)，但至少留 min_train_per_class 给 train
            target = int(round(n * val_ratio))
            max_val = max(0, n - min_train_per_class)
            k = min(max(0, target), max_val)

            # 为了 eval 稳一点：n>=2 时 k 至少 1（除非比例非常小且你不想 eval）
            if k == 0 and max_val > 0 and val_ratio > 0:
                k = 1

            val_indices.extend(cls_idx[:k].tolist())
            train_indices.extend(cls_idx[k:].tolist())

    # 去重 + 转 array
    val_idx = np.array(sorted(set(val_indices)), dtype=np.int64)
    train_idx = np.array(sorted(set(train_indices)), dtype=np.int64)

    # 防御：确保两边不重叠，且覆盖所有样本（若有空类等原因可能缺失，补回 train）
    used = set(val_idx.tolist()) | set(train_idx.tolist())
    if len(used) != n_total:
        missing = sorted(set(range(n_total)) - used)
        # 缺失的统一放 train
        train_idx = np.array(sorted(set(train_idx.tolist() + missing)), dtype=np.int64)

    inter = set(val_idx.tolist()) & set(train_idx.tolist())
    if inter:
        # 理论不该发生；发生则把交集挪到 train（更安全）
        val_set = set(val_idx.tolist())
        for i in inter:
            val_set.discard(i)
        val_idx = np.array(sorted(val_set), dtype=np.int64)

    return train_idx, val_idx


def get_kana(
    args,
    alg,
    name,
    num_labels,
    num_classes,
    data_dir,
    include_lb_to_ulb=True,
):
    """
    仿照 get_cifar 的接口
    返回:
        lb_dset, ulb_dset, eval_dset
    """

    # ------------------------------------------------------------------
    # 1) 收集文件
    # ------------------------------------------------------------------
    all_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    if len(all_paths) == 0:
        raise RuntimeError(f"No jpg found in {data_dir}")

    # ------------------------------------------------------------------
    # 2) 解析标签
    # ------------------------------------------------------------------
    all_targets = np.array(
        [parse_label(os.path.basename(p)) for p in all_paths],
        dtype=np.int64,
    )

    if all_targets.min() < 0 or all_targets.max() >= num_classes:
        raise ValueError(
            f"labels out of range: "
            f"min={all_targets.min()}, "
            f"max={all_targets.max()}, "
            f"num_classes={num_classes}"
        )

    # ------------------------------------------------------------------
    # 3) 分层划分 eval / train（不再严格 val_per_class）
    # ------------------------------------------------------------------
    seed = getattr(args, "seed", 0)
    rng = np.random.RandomState(seed)

    # 兼容：你可以继续传 args.val_per_class；但不再强制每类必须够
    # 若你想改成按比例：设置 args.val_per_class=None 且 args.val_ratio=0.1 等
    val_per_class = getattr(args, "val_per_class", None)
    val_ratio = getattr(args, "val_ratio", None)

    train_idx, val_idx = _stratified_split_eval_train(
        all_paths=all_paths,
        all_targets=all_targets,
        num_classes=num_classes,
        rng=rng,
        val_per_class=val_per_class,
        val_ratio=val_ratio,
        min_train_per_class=getattr(args, "min_train_per_class", 1),
    )

    train_paths = np.array(all_paths, dtype=object)[train_idx]
    train_targets = all_targets[train_idx]

    val_paths = np.array(all_paths, dtype=object)[val_idx]
    val_targets = all_targets[val_idx]

    # 打印 eval 分布（方便你看是否某些类 eval 为 0）
    eval_count = [0 for _ in range(num_classes)]
    for c in val_targets:
        eval_count[int(c)] += 1
    print(f"eval count: {eval_count} (total={len(val_targets)})")
    print(f"train size: {len(train_targets)}")

    # ------------------------------------------------------------------
    # 4) 读成 numpy（与 CIFAR 风格一致）
    # ------------------------------------------------------------------
    train_data = load_images_as_numpy(train_paths, args.img_size)
    # eval 可能为空，需兼容
    if len(val_paths) > 0:
        val_data = load_images_as_numpy(val_paths, args.img_size)
    else:
        # 空 eval：构造空数组占位，避免 BasicDataset 崩
        val_data = np.empty((0, args.img_size, args.img_size, 3), dtype=np.uint8)

    # ------------------------------------------------------------------
    # 5) 在 train 内部切分 lb / ulb
    # ------------------------------------------------------------------
    (
        lb_data,
        lb_targets,
        ulb_data,
        ulb_targets,
    ) = split_ssl_data(
        args,
        train_data,
        train_targets,
        num_classes,
        lb_num_labels=num_labels,
        ulb_num_labels=getattr(args, "ulb_num_labels", None),
        lb_imbalance_ratio=getattr(args, "lb_imb_ratio", 1.0),
        ulb_imbalance_ratio=getattr(args, "ulb_imb_ratio", 1.0),
        include_lb_to_ulb=include_lb_to_ulb,
    )

    # 打印分布（模仿 CIFAR）
    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]

    for c in lb_targets:
        lb_count[int(c)] += 1
    for c in ulb_targets:
        ulb_count[int(c)] += 1

    print(f"lb count:  {lb_count}")
    print(f"ulb count: {ulb_count}")

    # ------------------------------------------------------------------
    # 6) 数据增强
    # ------------------------------------------------------------------
    crop_size = args.img_size
    crop_ratio = getattr(args, "crop_ratio", 0.875)
    padding = int(crop_size * (1 - crop_ratio))

    base_aug = [
        transforms.Resize(crop_size),
        transforms.RandomCrop(
            crop_size,
            padding=padding,
            padding_mode="reflect",
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
    ]

    normalize = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )

    transform_weak = transforms.Compose([
        *base_aug,
        transforms.ToTensor(),
        normalize,
    ])

    transform_medium = transforms.Compose([
        *base_aug,
        RandAugment(1, 5),
        transforms.ToTensor(),
        normalize,
    ])

    transform_strong = transforms.Compose([
        *base_aug,
        RandAugment(3, 5),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        normalize,
    ])

    # ------------------------------------------------------------------
    # 7) 构建 Dataset
    # ------------------------------------------------------------------
    lb_dset = BasicDataset(
        alg,
        lb_data,
        lb_targets,
        num_classes,
        transform_weak,
        False,
        transform_strong,
        transform_strong,
        False,
    )

    ulb_dset = BasicDataset(
        alg,
        ulb_data,
        ulb_targets,
        num_classes,
        transform_weak,
        True,
        transform_medium,
        transform_strong,
        False,
    )

    eval_dset = BasicDataset(
        alg,
        val_data,
        val_targets,
        num_classes,
        transform_val,
        False,
        None,
        None,
        False,
    )

    return lb_dset, ulb_dset, eval_dset
