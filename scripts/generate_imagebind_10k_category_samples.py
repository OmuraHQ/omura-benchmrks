"""Generate 10k text-only category samples for embedding experiments.

This script creates an ImageBind-style prompt bank that can be used as
zero-shot anchors, nearest-neighbor probes, or atlas labels.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


GROUPS: dict[str, list[str]] = {
    "animal": [
        "cat",
        "dog",
        "bird",
        "horse",
        "cow",
        "sheep",
        "goat",
        "lion",
        "tiger",
        "elephant",
        "monkey",
        "rabbit",
        "deer",
        "fox",
        "wolf",
        "bear",
        "zebra",
        "camel",
        "penguin",
        "fish",
    ],
    "human": [
        "portrait",
        "selfie",
        "group photo",
        "person running",
        "person walking",
        "person smiling",
        "crowd",
        "family photo",
        "child",
        "elderly person",
        "worker",
        "athlete",
        "musician",
        "chef",
        "doctor",
        "student",
        "teacher",
        "dancer",
        "actor",
        "model",
    ],
    "nature": [
        "mountain",
        "forest",
        "beach",
        "river",
        "waterfall",
        "desert",
        "sunset",
        "sunrise",
        "clouds",
        "rain",
        "snow",
        "lake",
        "ocean",
        "valley",
        "island",
        "canyon",
        "jungle",
        "flower field",
        "tree",
        "skyline nature",
    ],
    "building": [
        "house",
        "apartment",
        "skyscraper",
        "temple",
        "church",
        "mosque",
        "castle",
        "bridge",
        "stadium",
        "library",
        "office building",
        "warehouse",
        "airport terminal",
        "train station",
        "museum",
        "school building",
        "hospital building",
        "mall",
        "factory",
        "city street",
    ],
    "vehicle": [
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
        "train",
        "airplane",
        "helicopter",
        "boat",
        "ship",
        "subway",
        "taxi",
        "ambulance",
        "police car",
        "tractor",
        "van",
        "scooter",
        "yacht",
        "race car",
        "pickup truck",
    ],
    "food": [
        "pizza",
        "burger",
        "pasta",
        "salad",
        "rice bowl",
        "sushi",
        "steak",
        "sandwich",
        "cake",
        "ice cream",
        "coffee",
        "tea",
        "soup",
        "noodles",
        "fruits",
        "vegetables",
        "dessert",
        "breakfast plate",
        "street food",
        "restaurant meal",
    ],
    "document": [
        "book page",
        "newspaper",
        "invoice",
        "receipt",
        "spreadsheet",
        "presentation slide",
        "report page",
        "research paper",
        "whiteboard notes",
        "chart",
        "table",
        "code snippet",
        "article page",
        "menu card",
        "poster text",
        "handwritten notes",
        "form document",
        "contract page",
        "resume",
        "manual page",
    ],
    "screen_ui": [
        "mobile app screen",
        "web dashboard",
        "login page",
        "chat interface",
        "code editor",
        "terminal screenshot",
        "video player ui",
        "settings page",
        "ecommerce page",
        "social media feed",
        "map interface",
        "analytics dashboard",
        "music app ui",
        "email client",
        "calendar screen",
        "profile page",
        "notification panel",
        "search page",
        "payment screen",
        "admin panel",
    ],
    "art": [
        "digital painting",
        "oil painting",
        "watercolor",
        "illustration",
        "comic art",
        "anime art",
        "pixel art",
        "sketch drawing",
        "poster design",
        "graffiti",
        "sculpture photo",
        "abstract art",
        "3d render",
        "concept art",
        "character art",
        "landscape painting",
        "portrait painting",
        "typography art",
        "logo design",
        "minimalist art",
    ],
    "meme": [
        "reaction meme",
        "captioned meme",
        "viral meme format",
        "funny screenshot",
        "text-overlay meme",
        "template meme",
        "internet joke image",
        "sarcastic meme",
        "relatable meme",
        "dank meme",
    ],
}


MODIFIERS = [
    "close-up",
    "wide shot",
    "high quality",
    "low light",
    "daytime",
    "nighttime",
    "outdoor",
    "indoor",
    "high contrast",
    "soft lighting",
    "vivid colors",
    "desaturated",
    "cinematic",
    "professional",
    "casual",
    "detailed",
    "minimal",
    "modern",
    "vintage",
    "realistic",
]


TEMPLATES = [
    "a photo of {label}",
    "an image of {label}",
    "a {modifier} photo of {label}",
    "{modifier} {label} scene",
    "{label} in a {modifier} style",
    "{label}, {modifier}",
    "{group} category: {label}",
    "{label} visual concept",
    "{label} sample for retrieval",
    "content showing {label}",
]


def build_samples(total: int, seed: int) -> list[dict[str, str]]:
    random.seed(seed)
    all_pairs: list[tuple[str, str]] = []
    for group, labels in GROUPS.items():
        for label in labels:
            all_pairs.append((group, label))

    out: list[dict[str, str]] = []
    for idx in range(total):
        group, label = random.choice(all_pairs)
        template = random.choice(TEMPLATES)
        modifier = random.choice(MODIFIERS)
        prompt = template.format(group=group, label=label, modifier=modifier)
        out.append(
            {
                "id": f"s{idx:05d}",
                "group": group,
                "label": label,
                "prompt": prompt,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 10k category text prompts")
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/category_samples/imagebind_10k_categories.jsonl"),
    )
    args = parser.parse_args()

    samples = build_samples(total=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for row in samples:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(samples)} samples to {args.output}")
    print(f"Groups: {', '.join(sorted(GROUPS.keys()))}")


if __name__ == "__main__":
    main()
