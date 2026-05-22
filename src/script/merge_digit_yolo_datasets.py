import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXTS = {".txt"}
SPLIT_NAMES = ["train", "val", "valid", "test"]
DEFAULT_NAMES = [str(i) for i in range(10)]
AUGMENT_OPS = ["photometric", "hflip", "vflip", "hflip_photometric", "vflip_photometric"]
AUG_RNG = np.random.default_rng()


def parse_simple_yaml(path: Path) -> Dict[str, object]:
    data = {}
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [item.strip().strip("'\"") for item in value[1:-1].split(",") if item.strip()]
            data[key] = items
        elif value.lower() in {"true", "false"}:
            data[key] = value.lower() == "true"
        elif value.isdigit():
            data[key] = int(value)
        else:
            data[key] = value.strip("'\"")
    return data


def infer_dataset_structure(root: Path) -> Dict[str, Tuple[Path, Path]]:
    result = {}
    if (root / "images" / "train").exists():
        for split in ["train", "val", "test"]:
            image_dir = root / "images" / split
            label_dir = root / "labels" / split
            if image_dir.exists():
                result[split] = (image_dir, label_dir)
        return result

    for split in ["train", "valid", "test"]:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if image_dir.exists():
            target_split = "val" if split == "valid" else split
            result[target_split] = (image_dir, label_dir)
    return result


def load_dataset_metadata(root: Path) -> Tuple[List[str], int]:
    yaml_file = None
    for candidate in ["data.yaml", "digit_yolo_data.yaml", "data.yml", "dataset.yaml"]:
        candidate_path = root / candidate
        if candidate_path.exists():
            yaml_file = candidate_path
            break

    if yaml_file is None:
        return DEFAULT_NAMES, len(DEFAULT_NAMES)

    meta = parse_simple_yaml(yaml_file)
    names = meta.get("names")
    nc = meta.get("nc")
    if names is None:
        names = DEFAULT_NAMES
    if nc is None:
        nc = len(names)
    return list(names), int(nc)


def collect_image_files(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def find_label_path(image_path: Path, image_dir: Path, label_dir: Path) -> Optional[Path]:
    stem = image_path.stem
    candidate = label_dir / f"{stem}.txt"
    if candidate.exists():
        return candidate
    # fallback for weird label naming conventions
    for ext in LABEL_EXTS:
        label_candidate = label_dir / f"{stem}{ext}"
        if label_candidate.exists():
            return label_candidate
    return None


def load_yolo_labels(label_path: Path) -> List[List[float]]:
    labels = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) != 5:
            continue
        labels.append([int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
    return labels


def write_yolo_labels(label_path: Path, labels: List[List[float]]) -> None:
    lines = [" ".join([str(int(lbl[0]))] + [f"{v:.6f}" for v in lbl[1:]]) for lbl in labels]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_label_file(label_path: Path, nc: int) -> bool:
    valid = True
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Invalid label format: {label_path} -> {line}")
            valid = False
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            print(f"Invalid class id in {label_path}: {parts[0]}")
            valid = False
            continue
        if cls < 0 or cls >= nc:
            print(f"Class id out of range in {label_path}: {cls} (nc={nc})")
            valid = False
    return valid


def make_unique_name(existing: set, name: str) -> str:
    if name not in existing:
        existing.add(name)
        return name
    base = Path(name).stem
    suffix = 1
    while True:
        candidate = f"{base}_{suffix}{Path(name).suffix}"
        if candidate not in existing:
            existing.add(candidate)
            return candidate
        suffix += 1


def flip_yolo_labels(labels: List[List[float]], flip_x: bool, flip_y: bool) -> List[List[float]]:
    if not flip_x and not flip_y:
        return labels
    flipped = []
    for cls, x, y, w, h in labels:
        if flip_x:
            x = 1.0 - x
        if flip_y:
            y = 1.0 - y
        flipped.append([cls, x, y, w, h])
    return flipped


def apply_random_photometric(image: np.ndarray) -> np.ndarray:
    result = image.copy().astype(np.float32)
    alpha = 1.0 + AUG_RNG.uniform(-0.3, 0.3)
    beta = AUG_RNG.uniform(-25, 25)
    result = result * alpha + beta
    result = np.clip(result, 0, 255).astype(np.uint8)

    if AUG_RNG.random() < 0.5:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + AUG_RNG.uniform(-10, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + AUG_RNG.uniform(-0.2, 0.2)), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + AUG_RNG.uniform(-0.2, 0.2)), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if AUG_RNG.random() < 0.35:
        ksize = int(AUG_RNG.choice([1, 3, 5]))
        if ksize > 1:
            result = cv2.GaussianBlur(result, (ksize, ksize), 0)

    if AUG_RNG.random() < 0.35:
        noise = AUG_RNG.normal(0, 12, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result


def augment_image_and_labels(image: np.ndarray, labels: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
    op = AUG_RNG.choice(AUGMENT_OPS)

    if op == "hflip":
        return cv2.flip(image, 1), flip_yolo_labels(labels, flip_x=True, flip_y=False)
    if op == "vflip":
        return cv2.flip(image, 0), flip_yolo_labels(labels, flip_x=False, flip_y=True)
    if op == "hflip_photometric":
        return apply_random_photometric(cv2.flip(image, 1)), flip_yolo_labels(labels, flip_x=True, flip_y=False)
    if op == "vflip_photometric":
        return apply_random_photometric(cv2.flip(image, 0)), flip_yolo_labels(labels, flip_x=False, flip_y=True)
    return apply_random_photometric(image), labels


def merge_datasets(dataset_roots: List[Path], out_root: Path, augment_roots: List[Path], augment_factor: int, dry_run: bool = False) -> None:
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    if not dry_run:
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)
    merged_splits = {}
    base_names: Optional[List[str]] = None
    base_nc: Optional[int] = None
    used_image_names = set()
    used_label_names = set()
    augment_roots_set = {root.resolve() for root in augment_roots}

    def should_augment(root: Path) -> bool:
        return root.resolve() in augment_roots_set

    for idx, root in enumerate(dataset_roots):
        structure = infer_dataset_structure(root)
        if not structure:
            raise ValueError(f"Cannot infer dataset structure for {root}")

        names, nc = load_dataset_metadata(root)
        if base_names is None:
            base_names = names
            base_nc = nc
        elif names != base_names or nc != base_nc:
            raise ValueError(
                f"Dataset metadata mismatch: {root}\n  names={names}\n  nc={nc}\n  expected names={base_names}\n  expected nc={base_nc}"
            )

        for split, (image_dir, label_dir) in structure.items():
            target_split = split
            target_image_dir = out_images / target_split
            target_label_dir = out_labels / target_split
            if not dry_run:
                target_image_dir.mkdir(parents=True, exist_ok=True)
                target_label_dir.mkdir(parents=True, exist_ok=True)
            merged_splits.setdefault(target_split, 0)

            if not image_dir.exists():
                continue
            augment_this_root = should_augment(root) and augment_factor > 0
            for image_path in collect_image_files(image_dir):
                label_path = find_label_path(image_path, image_dir, label_dir)
                if label_path is None:
                    print(f"Warning: no label found for image {image_path}")
                    continue
                valid = validate_label_file(label_path, base_nc)
                if not valid:
                    raise ValueError(f"Invalid label file: {label_path}")

                dest_image_name = make_unique_name(used_image_names, f"{idx}_{image_path.name}")
                dest_label_name = make_unique_name(used_label_names, f"{idx}_{label_path.name}")
                if not dry_run:
                    shutil.copy2(image_path, out_images / target_split / dest_image_name)
                    shutil.copy2(label_path, out_labels / target_split / dest_label_name)
                merged_splits[target_split] += 1

                if augment_this_root:
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise RuntimeError(f"Failed to read image for augmentation: {image_path}")
                    labels = load_yolo_labels(label_path)
                    for aug_index in range(augment_factor):
                        aug_image, aug_labels = augment_image_and_labels(image, labels)
                        aug_image_name = make_unique_name(
                            used_image_names,
                            f"{idx}_{image_path.stem}_aug{aug_index}{image_path.suffix}",
                        )
                        aug_label_name = make_unique_name(
                            used_label_names,
                            f"{idx}_{label_path.stem}_aug{aug_index}.txt",
                        )
                        if not dry_run:
                            cv2.imwrite(str(out_images / target_split / aug_image_name), aug_image)
                            write_yolo_labels(out_labels / target_split / aug_label_name, aug_labels)
                        merged_splits[target_split] += 1

    if base_names is None or base_nc is None:
        raise ValueError("No dataset metadata found in any source dataset")

    yaml_lines = [f"path: {out_root.as_posix()}"]
    for split in ["train", "val", "test"]:
        if split in merged_splits:
            yaml_lines.append(f"{split}: images/{split}")
    yaml_lines.append(f"nc: {base_nc}")
    yaml_lines.append("names: ['" + "','".join(base_names) + "']")

    if not dry_run:
        out_root.joinpath("data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print("Merged dataset summary:")
    for split, count in merged_splits.items():
        print(f"  {split}: {count} images")
    print(f"Output root: {out_root}")
    if not dry_run:
        print(f"Generated YAML: {out_root / 'data.yaml'}")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple YOLO digit datasets into one unified dataset")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Root folders of datasets to merge")
    parser.add_argument("--out_dir", required=True,
                        help="Destination root for merged dataset")
    parser.add_argument("--augment_sources", nargs="*", default=[],
                        help="Dataset roots whose images will be augmented during merge")
    parser.add_argument("--augment_factor", type=int, default=0,
                        help="Number of augmented copies to generate per source image for augmented datasets")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only validate and print summary without copying files")
    args = parser.parse_args()

    dataset_roots = [Path(p).expanduser().resolve() for p in args.datasets]
    augment_sources = [Path(p).expanduser().resolve() for p in args.augment_sources]
    out_root = Path(args.out_dir).expanduser().resolve()

    merge_datasets(dataset_roots, out_root, augment_sources, args.augment_factor, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
