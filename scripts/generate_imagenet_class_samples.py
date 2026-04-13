"""Generate category samples from ImageNet 1K class maps.

Input format: markdown table with columns:
| Class ID | Class Name |
| 0 | tench, Tinca tinca |
...
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.*?)\s*\|$")

TEMPLATES = [
    "a photo of {label}",
    "an image of {label}",
    "a close-up photo of {label}",
    "{label} in natural scene",
    "{label} object category",
    "{label}, high quality",
]


def parse_imagenet_markdown(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            match = ROW_RE.match(line.strip())
            if not match:
                continue
            class_id = int(match.group(1))
            class_name = match.group(2).strip()
            if class_name.lower() in {"class name", "---"}:
                continue
            if class_id < 0 or class_id > 999:
                continue
            primary_label = class_name.split(",")[0].strip()
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "primary_label": primary_label,
                }
            )

    rows.sort(key=lambda x: x["class_id"])
    return rows


def build_expanded_samples(
    classes: list[dict[str, str]],
    count: int,
    seed: int,
) -> list[dict[str, str]]:
    random.seed(seed)
    out: list[dict[str, str]] = []
    for idx in range(count):
        cls = random.choice(classes)
        template = random.choice(TEMPLATES)
        prompt = template.format(label=cls["primary_label"])
        out.append(
            {
                "id": f"imgnet_s{idx:05d}",
                "source": "imagenet_1k",
                "class_id": cls["class_id"],
                "class_name": cls["class_name"],
                "label": cls["primary_label"],
                "prompt": prompt,
            }
        )
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ImageNet class samples")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/root/.cursor/projects/workspace-proj-omura/uploads/IMAGENET-0.md"),
        help="Path to ImageNet markdown class-map file",
    )
    parser.add_argument(
        "--output-classes",
        type=Path,
        default=Path("data/category_samples/imagenet_1k_classes.jsonl"),
        help="Output JSONL for raw 1k classes",
    )
    parser.add_argument(
        "--output-prompts",
        type=Path,
        default=Path("data/category_samples/imagenet_10k_prompt_samples.jsonl"),
        help="Output JSONL for expanded prompt samples",
    )
    parser.add_argument(
        "--prompt-count",
        type=int,
        default=10_000,
        help="How many expanded prompt rows to generate",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    classes = parse_imagenet_markdown(args.input)
    if len(classes) != 1000:
        raise RuntimeError(f"Expected 1000 classes, found {len(classes)} from {args.input}")

    class_rows = []
    for cls in classes:
        class_rows.append(
            {
                "source": "imagenet_1k",
                "class_id": cls["class_id"],
                "class_name": cls["class_name"],
                "label": cls["primary_label"],
                "prompt": f"a photo of {cls['primary_label']}",
            }
        )

    prompt_rows = build_expanded_samples(
        classes=classes,
        count=args.prompt_count,
        seed=args.seed,
    )

    write_jsonl(args.output_classes, class_rows)
    write_jsonl(args.output_prompts, prompt_rows)

    print(f"Loaded classes: {len(classes)}")
    print(f"Wrote raw classes: {len(class_rows)} -> {args.output_classes}")
    print(f"Wrote expanded prompts: {len(prompt_rows)} -> {args.output_prompts}")


if __name__ == "__main__":
    main()
