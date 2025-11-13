from __future__ import annotations

import random
import shutil
from pathlib import Path

from ultralytics.utils.files import increment_path  # helpful to create a new dir


def split_yolo_dataset(source_dir: str | Path, train_ratio: float = 0.8) -> Path:
    """Split YOLO segmentation dataset into train and val directories in a new directory.

    Args:
        source_dir: Path to YOLO dataset root directory containing ``images`` and ``labels`` subfolders.
        train_ratio: Ratio for train split, between 0 and 1. Defaults to 0.8.

    Returns:
        Path: Path to the generated split dataset directory.

    Raises:
        FileNotFoundError: If the dataset root or required files are missing.
        ValueError: If the dataset structure is invalid or the split ratio is out of range.
    """
    src = Path(source_dir).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Dataset root not found: {src}")

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")

    images_dir = src / "images"
    labels_dir = src / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise ValueError("source_dir must contain 'images' and 'labels' subdirectories.")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = sorted(
        p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_extensions
    )
    if not image_paths:
        raise ValueError("No image files found in source_dir/images.")

    dataset_pairs: list[tuple[Path, Path]] = []
    missing_labels: list[str] = []
    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.is_file():
            dataset_pairs.append((image_path, label_path))
        else:
            missing_labels.append(image_path.name)

    if missing_labels:
        missing_preview = ", ".join(missing_labels[:5])
        suffix = "..." if len(missing_labels) > 5 else ""
        raise FileNotFoundError(
            f"Missing label files for {len(missing_labels)} image(s): {missing_preview}{suffix}"
        )

    if not dataset_pairs:
        raise ValueError("No matching image/label pairs found in the dataset.")

    random.shuffle(dataset_pairs)

    train_size = int(len(dataset_pairs) * train_ratio)
    if train_size == 0 or train_size == len(dataset_pairs):
        raise ValueError(
            "train_ratio results in an empty train or val split; adjust the ratio or add more data."
        )

    train_pairs = dataset_pairs[:train_size]
    val_pairs = dataset_pairs[train_size:]

    dst = src.with_name(f"{src.name}_split")
    images_train_dir = dst / "images" / "train"
    images_val_dir = dst / "images" / "val"
    labels_train_dir = dst / "labels" / "train"
    labels_val_dir = dst / "labels" / "val"

    for directory in (images_train_dir, images_val_dir, labels_train_dir, labels_val_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs: list[tuple[Path, Path]], image_dst: Path, label_dst: Path) -> None:
        for image_path, label_path in pairs:
            shutil.copy2(image_path, image_dst / image_path.name)
            shutil.copy2(label_path, label_dst / label_path.name)

    copy_pairs(train_pairs, images_train_dir, labels_train_dir)
    copy_pairs(val_pairs, images_val_dir, labels_val_dir)

    return dst
