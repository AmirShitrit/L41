from pathlib import Path

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _list_images(class_dir):
    return [f for f in class_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS]


def _class_sample_counts(data_dir):
    class_dirs = sorted(p for p in Path(data_dir).iterdir() if p.is_dir())
    return [(d.name, len(_list_images(d))) for d in class_dirs]


def _sample_image_attributes(data_dir, max_per_class=50):
    widths, heights, modes = [], [], set()
    for class_dir in Path(data_dir).iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in _list_images(class_dir)[:max_per_class]:
            with Image.open(img_path) as img:
                widths.append(img.width)
                heights.append(img.height)
                modes.add(img.mode)
    return widths, heights, list(modes)


def _balance_label(ratio):
    if ratio > 0.8:
        return "balanced"
    if ratio > 0.5:
        return "mildly imbalanced"
    return "imbalanced"


def print_dataset_stats(data_dir):
    counts = _class_sample_counts(data_dir)
    widths, heights, modes = _sample_image_attributes(data_dir)

    total = sum(c for _, c in counts)
    sample_counts = [c for _, c in counts]
    balance_ratio = min(sample_counts) / max(sample_counts)
    max_count = max(sample_counts)

    print("=== Dataset Statistics ===")
    print(f"\nClasses ({len(counts)}):")
    for name, count in counts:
        bar = "█" * round(20 * count / max_count)
        print(f"  {name:<20} {count:>5} samples  {bar}")

    print(f"\nTotal samples : {total}")
    print(f"Balance ratio : {balance_ratio:.3f}  ({_balance_label(balance_ratio)})")
    print(f"  min={min(sample_counts)}, max={max_count}, "
          f"mean={total / len(counts):.0f} samples/class")

    print(f"\nImage attributes (sampled {len(widths)} images):")
    print(f"  Width  — min={min(widths)}, max={max(widths)}, mean={sum(widths) / len(widths):.0f} px")
    print(f"  Height — min={min(heights)}, max={max(heights)}, mean={sum(heights) / len(heights):.0f} px")
    print(f"  Color modes present: {', '.join(sorted(modes))}")
    print()
