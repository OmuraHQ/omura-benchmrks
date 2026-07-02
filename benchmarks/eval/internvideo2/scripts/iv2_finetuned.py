"""Load the finetuned 'omura embed video' model (base InternVideo2-6B + finetuned heads)
and expose text/video embedding in the finetuned joint space (768-d, L2-normalized).

Shared by the indexing backfill (index_video_iv2.py) and the inference microservice
(iv2_video_service.py). Runs in .venv-iv2 (transformers 4.28 + repo code).
"""
import os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import iv2_common as C

_state = {"model": None, "tok": None, "cfg": None}


def load(heads_path: str | None = None, device: str = "cuda"):
    """Load base model and apply finetuned heads (vision_proj/text_proj/text_encoder/temp)."""
    if _state["model"] is not None:
        return _state["model"], _state["tok"], _state["cfg"]
    model, tok, cfg = C.load_model(device=device)
    heads_path = heads_path or os.getenv("OMURA_VIDEO_HEADS", "")
    if heads_path and os.path.exists(heads_path):
        ckpt = torch.load(heads_path, map_location="cpu", weights_only=False)
        # heads were trained in fp32; cast to the model dtype for inference
        model.text_encoder.float().load_state_dict(ckpt["text_encoder"])
        model.vision_proj.float().load_state_dict(ckpt["vision_proj"])
        model.text_proj.float().load_state_dict(ckpt["text_proj"])
        if "temp" in ckpt:
            model.temp.data = ckpt["temp"].to(model.temp.device).float()
        print(f"[omura-embed-video] applied finetuned heads from {heads_path} "
              f"(epoch={ckpt.get('epoch')}, metrics={ckpt.get('metrics')})")
    else:
        print("[omura-embed-video] WARNING: no finetuned heads applied (zero-shot base).")
    model.eval()
    _state.update(model=model, tok=tok, cfg=cfg)
    return model, tok, cfg


@torch.no_grad()
def embed_text(texts, device="cuda") -> np.ndarray:
    model, tok, cfg = load(device=device)
    if isinstance(texts, str):
        texts = [texts]
    t = tok(list(texts), padding="max_length", truncation=True,
            max_length=cfg.max_txt_l, return_tensors="pt").to(device)
    _, pooled = model.encode_text(t)
    feat = F.normalize(model.text_proj(pooled.float()).float(), dim=-1)
    return feat.cpu().numpy()


@torch.no_grad()
def embed_video_frames(frames_tensor, device="cuda") -> np.ndarray:
    """frames_tensor [B,T,C,H,W] float (from iv2_common.read_video_official)."""
    model, tok, cfg = load(device=device)
    frames_tensor = frames_tensor.to(model._iv2_dtype)
    _, pooled = model.encode_vision(frames_tensor, test=True)
    if pooled.dim() == 3:
        pooled = pooled.squeeze(1)
    feat = F.normalize(model.vision_proj(pooled.float()).float(), dim=-1)
    return feat.cpu().numpy()
