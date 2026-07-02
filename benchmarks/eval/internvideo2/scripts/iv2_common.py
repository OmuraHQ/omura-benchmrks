"""Shared model loading + feature extraction for InternVideo2-6B (audiovisual)."""
import os, sys
import numpy as np
import torch

# Import the repo as the `multi_modality` package so that package-relative
# imports (e.g. `from ..utils.distributed import ...`) resolve correctly.
_MM = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "repo", "InternVideo2", "multi_modality"))
_PARENT = os.path.dirname(_MM)  # repo/InternVideo2
for p in (_PARENT, _MM):
    if p not in sys.path:
        sys.path.insert(0, p)

CKPT = os.environ["IV2_CKPT"]

from multi_modality.utils.config import Config, eval_dict_leaf  # noqa: E402
from multi_modality.models.backbones.bert.tokenization_bert import BertTokenizer  # noqa: E402
from multi_modality.models.internvideo2_stage2_audiovisual import InternVideo2_Stage2_audiovisual  # noqa: E402
from multi_modality.models.backbones.internvideo2.pos_embed import (  # noqa: E402
    interpolate_pos_embed_internvideo2_new,
)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "iv2_6b_av_config.py")

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def _normalize(data):
    return (data / 255.0 - v_mean) / v_std


def read_video_official(video_path, num_frames=4, image_res=224, sample="middle", device="cuda"):
    """Match the repo's test pipeline: decord uniform-interval (middle) frame
    indices + BICUBIC Resize((res,res)) + /255 + ImageNet-video normalize.
    Returns [1,T,C,H,W] float on `device`, or None if unreadable."""
    import decord
    from decord import VideoReader
    import torch as _t
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    decord.bridge.set_bridge("torch")
    try:
        vr = VideoReader(video_path, num_threads=1)
    except Exception:
        return None
    vlen = len(vr)
    if vlen == 0:
        return None
    # official get_frame_indices (middle): split into num_frames intervals, take midpoints
    acc = min(num_frames, vlen)
    intervals = np.linspace(0, vlen, acc + 1).astype(int)
    idx = [(intervals[i] + intervals[i + 1] - 1) // 2 for i in range(acc)]
    if len(idx) < num_frames:
        idx = idx + [idx[-1]] * (num_frames - len(idx))
    frames = vr.get_batch(idx)  # [T,H,W,C] uint8 (torch)
    frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]
    tfm = transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    frames = tfm(frames)  # [T,C,H,W]
    return frames.unsqueeze(0).to(device)  # [1,T,C,H,W]


def video_duration(video_path):
    """Return (duration_seconds, fps, num_frames) for a video, or (None, None, None)."""
    import decord
    from decord import VideoReader
    try:
        vr = VideoReader(video_path, num_threads=1)
    except Exception:
        return None, None, None
    vlen = len(vr)
    if vlen == 0:
        return None, None, None
    fps = float(vr.get_avg_fps()) or 25.0
    return vlen / fps, fps, vlen


def read_video_window(video_path, start_sec, end_sec, num_frames=4, image_res=224, device="cuda", _vr=None):
    """Sample `num_frames` evenly within the time window [start_sec, end_sec] of a video,
    using the same BICUBIC + ImageNet-video normalization as read_video_official.
    Returns [1,T,C,H,W] float on `device`, or None if unreadable. Pass `_vr` to reuse a
    decord VideoReader across many windows of the same clip (avoids re-opening the file)."""
    import decord
    from decord import VideoReader
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    decord.bridge.set_bridge("torch")
    vr = _vr
    if vr is None:
        try:
            vr = VideoReader(video_path, num_threads=1)
        except Exception:
            return None
    vlen = len(vr)
    if vlen == 0:
        return None
    fps = float(vr.get_avg_fps()) or 25.0
    s = max(0, int(round(start_sec * fps)))
    e = min(vlen - 1, int(round(end_sec * fps)))
    if e <= s:
        e = min(vlen - 1, s + 1)
    idx = np.linspace(s, e, num_frames).astype(int).tolist()
    frames = vr.get_batch(idx)  # [T,H,W,C] uint8 (torch)
    frames = frames.permute(0, 3, 1, 2)  # [T,C,H,W]
    tfm = transforms.Compose([
        transforms.Resize((image_res, image_res), interpolation=InterpolationMode.BICUBIC),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    frames = tfm(frames)  # [T,C,H,W]
    return frames.unsqueeze(0).to(device)  # [1,T,C,H,W]


def frames2tensor(vid_list, fnum=4, target_size=(224, 224), device="cuda"):
    """vid_list: list of HxWx3 BGR frames (opencv). Returns [1,T,C,H,W]."""
    import cv2
    assert len(vid_list) >= fnum, f"need >= {fnum} frames, got {len(vid_list)}"
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]  # BGR->RGB
    vid_tube = [np.expand_dims(_normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))  # [1,T,C,H,W]
    return torch.from_numpy(vid_tube).to(device, non_blocking=True).float()


def load_model(device="cuda"):
    cfg = Config.from_file(CONFIG_PATH)
    cfg = eval_dict_leaf(cfg)
    cfg.pretrained_path = CKPT
    cfg.model.audio_encoder.audio_model_path = os.path.abspath(
        cfg.model.audio_encoder.audio_model_path)

    # The repo references config files (e.g. configs/config_bert_large.json) by
    # relative path, so build the model with cwd at the repo root.
    _prev_cwd = os.getcwd()
    os.chdir(_MM)
    try:
        tokenizer = BertTokenizer.from_pretrained(cfg.model.text_encoder.pretrained)
        model = InternVideo2_Stage2_audiovisual(config=cfg, tokenizer=tokenizer, is_pretrain=True)
    finally:
        os.chdir(_prev_cwd)

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if "model" in ckpt else ckpt.get("module", ckpt)
    # interpolate temporal pos embed (ckpt trained with f4 already, orig 4)
    try:
        interpolate_pos_embed_internvideo2_new(sd, model.vision_encoder, orig_t_size=4)
    except Exception as e:
        print("pos-embed interp skipped:", e)
    msg = model.load_state_dict(sd, strict=False)
    miss = [k for k in msg.missing_keys if not k.startswith("vision_encoder.clip")]
    print(f"[load] missing(non-clip-teacher)={len(miss)} unexpected={len(msg.unexpected_keys)}")
    if miss[:10]:
        print("  sample missing:", miss[:10])

    dtype = torch.bfloat16 if cfg.get("use_bf16", False) else (
        torch.float16 if cfg.get("use_half_precision", False) else torch.float32)
    model = model.to(device).to(dtype)
    model.eval()
    model._iv2_dtype = dtype

    # Optionally apply finetuned 'omura embed video' heads (text_encoder/projections/temp).
    heads = os.environ.get("OMURA_VIDEO_HEADS", "")
    if heads and os.path.exists(heads):
        hc = torch.load(heads, map_location="cpu", weights_only=False)
        model.text_encoder.load_state_dict({k: v.to(dtype) for k, v in hc["text_encoder"].items()})
        model.vision_proj.load_state_dict({k: v.to(dtype) for k, v in hc["vision_proj"].items()})
        model.text_proj.load_state_dict({k: v.to(dtype) for k, v in hc["text_proj"].items()})
        if "temp" in hc:
            model.temp.data = hc["temp"].to(device).to(dtype)
        print(f"[load] applied finetuned heads from {heads} (epoch={hc.get('epoch')}, {hc.get('metrics')})")

    return model, tokenizer, cfg


@torch.no_grad()
def get_video_feat(model, frames_tensor):
    """frames_tensor [B,T,C,H,W] float. Returns L2-normalized contrastive vision feat [B,768]."""
    frames_tensor = frames_tensor.to(model._iv2_dtype)
    _, pooled = model.encode_vision(frames_tensor, test=True)  # pooled [B,768]
    if pooled.dim() == 3:
        pooled = pooled.squeeze(1)
    vfeat = model.vision_proj(pooled)
    vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
    return vfeat.float()


@torch.no_grad()
def get_text_feat(model, tokenizer, texts, max_txt_l=40, device="cuda"):
    """texts: list[str]. Returns L2-normalized contrastive text feat [N,768]."""
    tok = tokenizer(list(texts), padding="max_length", truncation=True,
                    max_length=max_txt_l, return_tensors="pt").to(device)
    _, pooled = model.encode_text(tok)  # pooled [N,1024]
    tfeat = model.text_proj(pooled)
    tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    return tfeat.float()


@torch.no_grad()
def get_audio_feat(model, waveforms_16k, device="cuda", max_seconds=10, sr=16000):
    """waveforms_16k: list of 1-D float tensors (mono, 16kHz).
    Each clip is pad/truncated to `max_seconds` (matches training max_audio_length=10)
    so fbanks share a time dimension and can be batched.
    Returns L2-normalized contrastive audio feat [N,768]."""
    target = int(max_seconds * sr)
    fbanks = []
    for w in waveforms_16k:
        w = w.to(device).float().flatten()
        if w.numel() < target:
            w = torch.nn.functional.pad(w, (0, target - w.numel()))
        else:
            w = w[:target]
        fb = model.audio_encoder.preprocess(w.unsqueeze(0))  # [1,T,128]
        fbanks.append(fb)
    fbank = torch.cat(fbanks, dim=0).to(model._iv2_dtype)  # [N,T,128]
    audio_embeds = model.audio_encoder(fbank)               # [N, tokens, 768]
    pooled = audio_embeds.mean(dim=1)                        # [N,768]
    afeat = model.audio_proj(pooled)
    afeat = afeat / afeat.norm(dim=-1, keepdim=True)
    return afeat.float()
