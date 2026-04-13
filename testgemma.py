#!/usr/bin/env python3
"""Quick encoder probe for google/gemma-4-E4B-it.

This script is intentionally lightweight for experimentation:
- Loads model + processor from Hugging Face
- Accepts text and/or image input
- Runs a forward pass with hidden states
- Mean-pools token states into one normalized embedding vector
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import soundfile as sf
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
)


DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"


def _pick_dtype(device: str) -> torch.dtype:
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _pool_last_hidden(hidden: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    # hidden: [batch, seq_len, hidden_size]
    if attention_mask is None:
        pooled = hidden.mean(dim=1)
    else:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1e-6)
        pooled = summed / denom
    return torch.nn.functional.normalize(pooled, p=2, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test Gemma-4 as an embedding encoder.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF model id")
    parser.add_argument("--text", default="a cute cat on a skateboard", help="Input text")
    parser.add_argument("--image", type=str, default=None, help="Optional image path")
    parser.add_argument("--audio", type=str, default=None, help="Optional audio path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-npy", type=str, default=None, help="Optional output .npy path")
    args = parser.parse_args()

    dtype = _pick_dtype(args.device)
    print(f"[GemmaTest] Loading processor/model: {args.model_id}")
    processor = None
    tokenizer = None
    image_processor = None
    try:
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    except Exception as e:
        print(f"[GemmaTest] AutoProcessor unavailable: {e}")
        print(
            "[GemmaTest] Your transformers build likely lacks Gemma-4 processor "
            "registration. Trying tokenizer/image-processor fallback."
        )
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        except Exception as te:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_id, trust_remote_code=True, use_fast=False
                )
            except Exception as te2:
                raise RuntimeError(
                    "Could not load tokenizer fallback either. This usually means the current "
                    "transformers/tokenizer stack is still incompatible with Gemma-4 metadata "
                    "(e.g. extra_special_tokens format mismatch). Upgrade to a newer "
                    "transformers build and retry."
                ) from te2
        try:
            image_processor = AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        except Exception:
            image_processor = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(args.device)
    model.eval()

    image = None
    if args.image:
        image = Image.open(args.image).convert("RGB")
    audio_data = None
    audio_sr = None
    if args.audio:
        audio_data, audio_sr = sf.read(args.audio, always_2d=False)
        if isinstance(audio_data, np.ndarray) and audio_data.ndim > 1:
            # Convert stereo/multi-channel to mono for simpler probing.
            audio_data = audio_data.mean(axis=1)

    # For multimodal models, include text even when image/audio are present.
    prompt = args.text.strip() or "describe the provided media"
    if processor is not None:
        # Gemma-4 expects media placeholder tokens in the text sequence.
        # Build a chat-style prompt so image/audio features align with tokens.
        proc_text = prompt
        proc_kwargs = {"return_tensors": "pt"}
        if image is not None or audio_data is not None:
            content = []
            if image is not None:
                content.append({"type": "image"})
            if audio_data is not None:
                content.append({"type": "audio"})
            content.append({"type": "text", "text": prompt})
            messages = [{"role": "user", "content": content}]
            proc_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        proc_kwargs["text"] = proc_text
        if image is not None:
            proc_kwargs["images"] = image
        if audio_data is not None:
            # Keep both key variants for compatibility across processor versions.
            proc_kwargs["audio"] = audio_data
            proc_kwargs["audios"] = [audio_data]
            proc_kwargs["sampling_rate"] = int(audio_sr)
        inputs = processor(**proc_kwargs)
    else:
        # Text fallback path when AutoProcessor class is not available.
        inputs = tokenizer(prompt, return_tensors="pt")
        if image is not None and image_processor is None:
            print(
                "[GemmaTest] Image was provided, but this transformers build cannot load "
                "Gemma image processor. Continuing with text-only fallback."
            )
        if audio_data is not None:
            print(
                "[GemmaTest] Audio was provided, but AutoProcessor is unavailable in this "
                "transformers build. Continuing with text-only fallback."
            )

    inputs = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)

    last_hidden = out.hidden_states[-1]  # [B, T, H]
    embedding = _pool_last_hidden(last_hidden, inputs.get("attention_mask"))
    vec = embedding[0].float().cpu().numpy()

    print(f"[GemmaTest] embedding_dim={vec.shape[0]}")
    print(f"[GemmaTest] l2_norm={np.linalg.norm(vec):.6f}")
    print(f"[GemmaTest] first_8={np.array2string(vec[:8], precision=5)}")

    if args.save_npy:
        out_path = Path(args.save_npy)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, vec)
        print(f"[GemmaTest] saved={out_path}")


if __name__ == "__main__":
    main()
