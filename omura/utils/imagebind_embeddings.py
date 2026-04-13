"""
Generates multimodal embeddings for text/images/video.
Default model: Omura Embed.

GPU parallelism: one model instance per visible CUDA device, served from a
thread-safe queue. Set CUDA_VISIBLE_DEVICES to control which GPUs are used.
"""

import os
import logging
import threading
import queue
import gc
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from PIL import Image
from transformers import AutoModel, AutoModelForImageTextToText, AutoProcessor


# Filter out "System prompt modified" and "Unrecognized keys in rope_scaling"
class _LogFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "System prompt modified" in msg:
            return False
        if "Unrecognized keys in " in msg and "rope_scaling" in msg:
            return False
        return True


_log_filter = _LogFilter()
logging.getLogger().addFilter(_log_filter)
logging.getLogger("transformers").addFilter(_log_filter)
logging.getLogger("transformers.modeling_rope_utils").addFilter(_log_filter)

MODEL_NAME = os.getenv("OMURA_EMBEDDING_MODEL", "immortaltatsu/omura_emmbed")

# Pool of (model, processor, device) tuples — one entry per GPU.
# Workers get() a slot, run inference, put() it back.
_model_pool: queue.Queue = queue.Queue()
_pool_size: int = 0  # number of loaded instances; 0 = not loaded yet

_load_lock = threading.Lock()
_LOAD_RETRY_SECONDS = float(os.getenv("OMURA_MODEL_RETRY_SECONDS", "120"))
_load_failed_at: float = 0.0  # monotonic time of last failure; 0 = never failed
_fallback_model = None
_fallback_lock = threading.Lock()

_QWEN_QUERY_INSTRUCTION = (
    "Retrieve images or text relevant to the user's query."
)
_QWEN_DOCUMENT_INSTRUCTION = "Represent this document for retrieval."
_QWEN_IMAGE_INSTRUCTION = "Retrieve images or text relevant to this image."


def _target_embedding_dim() -> int:
    return int(os.getenv("OMURA_EMBEDDING_DIM", "768"))


def _reshape_embedding_dim(vec: np.ndarray, dim: int) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).flatten()
    if arr.shape[0] == dim:
        out = arr
    elif arr.shape[0] > dim:
        out = arr[:dim]
    else:
        out = np.pad(arr, (0, dim - arr.shape[0]), mode="constant")
    n = np.linalg.norm(out)
    if n > 0:
        out = out / n
    return out.astype(np.float32, copy=False)


def _get_fallback_model():
    global _fallback_model
    if _fallback_model is not None:
        return _fallback_model
    with _fallback_lock:
        if _fallback_model is not None:
            return _fallback_model
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        m = AutoModel.from_pretrained(
            _FALLBACK_MODEL_NAME,
            trust_remote_code=True,
            dtype=dtype,
        )
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        m = m.to(dev)
        m.eval()
        _fallback_model = (m, dev)
    return _fallback_model


def _encode_fallback_text(texts: list[str]) -> Optional[np.ndarray]:
    try:
        model, _ = _get_fallback_model()
        with torch.no_grad():
            embs = model.encode_text(texts, normalize=True)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().float().numpy()
        embs = np.asarray(embs, dtype=np.float32)
        dim = _target_embedding_dim()
        return np.vstack([_reshape_embedding_dim(v, dim) for v in embs])
    except Exception as exc:
        print(f"[Embedding] Fallback text error: {exc}")
        return None


def _encode_fallback_images(images: list[Image.Image]) -> Optional[np.ndarray]:
    try:
        model, _ = _get_fallback_model()
        with torch.no_grad():
            embs = model.encode_image(images, normalize=True)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().float().numpy()
        embs = np.asarray(embs, dtype=np.float32)
        dim = _target_embedding_dim()
        return np.vstack([_reshape_embedding_dim(v, dim) for v in embs])
    except Exception as exc:
        print(f"[Embedding] Fallback image error: {exc}")
        return None


class _ModelNotReadyError(Exception):
    """Raised when the model pool is in backoff after a failed load."""


def _target_devices() -> list[str]:
    """Return configured CUDA device strings, or ['cpu'] if none.

    Controls:
    - OMURA_EMBEDDING_DEVICES: comma-separated explicit list (e.g. "cuda:0,cuda:1" or "0,1")
    - OMURA_EMBEDDING_MAX_DEVICES: cap auto-detected visible GPUs.
      - If <= 0, use all visible GPUs.
      - Default: 1
    """
    explicit = os.getenv("OMURA_EMBEDDING_DEVICES", "").strip()
    if explicit:
        out: list[str] = []
        for raw in explicit.split(","):
            token = raw.strip()
            if not token:
                continue
            if token.startswith("cuda:"):
                out.append(token)
            elif token.isdigit():
                out.append(f"cuda:{token}")
        if out:
            return out

    n = torch.cuda.device_count()
    if n == 0:
        return ["cpu"]
    max_devices = int(os.getenv("OMURA_EMBEDDING_MAX_DEVICES", "1"))
    if max_devices <= 0:
        max_devices = n
    max_devices = max(1, min(max_devices, n))
    return [f"cuda:{i}" for i in range(max_devices)]


def ensure_model_loaded() -> None:
    """Load model instances into the pool if not already loaded. Thread-safe."""
    global _pool_size
    if _pool_size > 0:
        return
    with _load_lock:
        if _pool_size > 0:
            return
        if _load_failed_at > 0:
            remaining = _LOAD_RETRY_SECONDS - (time.monotonic() - _load_failed_at)
            if remaining > 0:
                raise _ModelNotReadyError(
                    f"Model unavailable (load failed); retrying in {remaining:.0f}s"
                )
        _load_model()


def initialize_embedding_model() -> None:
    """Load embedding model pool at startup. Idempotent."""
    ensure_model_loaded()


# Backward-compatible alias for older call sites.
initialize_imagebind = initialize_embedding_model


def _is_gemma_model() -> bool:
    return "gemma-4" in MODEL_NAME.lower()


def _is_omura_emmbed_model() -> bool:
    """Return True when the active backend is Omura Embed."""
    backend = os.getenv("OMURA_EMBEDDING_BACKEND", "").strip().lower()
    if backend in {"omura_emmbed", "omura_emmbed_v1"}:
        return True
    name = MODEL_NAME.lower()
    return ("omura_emmbed" in name) and "jina" not in name


def _is_jina_clip_model() -> bool:
    """Return True for jinaai/jina-clip-* models (89-language CLIP, 1024-d)."""
    name = MODEL_NAME.lower()
    return "jina-clip" in name or "jina_clip" in name


def _is_qwen3_vl_embedding_model() -> bool:
    """Return True for Qwen3-VL embedding checkpoints."""
    name = MODEL_NAME.lower()
    return "qwen3-vl-embedding" in name


def _load_one(device: str, attn_impl: str) -> tuple:
    """Load a single (model, processor) onto *device* with the given attention impl.

    Returns (model, processor) where processor is None for models with built-in encode_* methods.
    """
    offload_dir = "data/model_offload"

    # ── Jina CLIP v2 ──
    if _is_jina_clip_model():
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        m = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            dtype=dtype,
        )
        m = m.to(device)
        m.eval()
        return m, None  # no separate processor

    # ── Omura Embed backend ──
    model_cls = AutoModelForImageTextToText if _is_gemma_model() else AutoModel
    if not _is_gemma_model() and not _is_omura_emmbed_model():
        if _is_qwen3_vl_embedding_model():
            try:
                from qwen_vl_utils import process_vision_info  # noqa: F401
            except ImportError:
                raise ImportError(
                    "qwen-vl-utils required. Install with `uv add qwen-vl-utils`"
                )
        else:
            try:
                import qwen_omni_utils  # noqa: F401
            except ImportError:
                raise ImportError(
                    "qwen-omni-utils required. Install with `uv add qwen-omni-utils`"
                )

    m = model_cls.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map=device,
        low_cpu_mem_usage=True,
        offload_folder=offload_dir,
        offload_buffers=True,
    )
    m.eval()
    if _is_omura_emmbed_model():
        # Omura Embed processors do not use Qwen-specific pixel bounds
        p = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    else:
        p = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
    return m, p


def _load_model() -> None:
    global _pool_size, _load_failed_at

    attn_impl = "eager"
    devices = _target_devices()
    print(f"[Embedding] Loading {MODEL_NAME} on {devices} (attn={attn_impl})…")

    loaded = 0
    for device in devices:
        for attempt_attn in ["eager"]:
            try:
                model, processor = _load_one(device, attempt_attn)
                _model_pool.put((model, processor, device))
                loaded += 1
                print(f"[Embedding] Loaded on {device} (attn={attempt_attn})")
                break
            except Exception as e:
                print(f"[Embedding] FAILED on {device}: {e}")

    if loaded == 0:
        _load_failed_at = time.monotonic()
        raise RuntimeError(f"Could not load {MODEL_NAME} on any device: {devices}")

    _pool_size = loaded
    print(f"[Embedding] Pool ready: {loaded}/{len(devices)} device(s)")


def _encode_documents(documents):
    """
    Encodes a list of 'documents'.
    Each document is a list of messages: [{"role": "user", "content": [...]}]

    Acquires a model slot from the pool, runs inference, returns the slot.
    """
    ensure_model_loaded()  # raises _ModelNotReadyError if in backoff

    # Acquire a free model instance (blocks until one is available)
    model, processor, device = _model_pool.get()
    try:
        doc = documents[0]
        documents_texts = processor.apply_chat_template(
            doc, add_generation_prompt=False, tokenize=False
        )

        processor_kw = {"text": documents_texts, "return_tensors": "pt"}
        if _is_gemma_model():
            images = []
            audios = []
            sampling_rate = None
            for message in doc:
                for part in message.get("content", []):
                    kind = part.get("type")
                    if kind == "image" and part.get("image") is not None:
                        images.append(part["image"])
                    elif kind == "audio" and part.get("audio") is not None:
                        arr, sr = sf.read(part["audio"], always_2d=False)
                        if isinstance(arr, np.ndarray) and arr.ndim > 1:
                            arr = arr.mean(axis=1)
                        audios.append(arr)
                        sampling_rate = int(sr)
            if images:
                processor_kw["images"] = images
            if audios:
                processor_kw["audios"] = audios
                processor_kw["sampling_rate"] = sampling_rate
        else:
            if isinstance(documents_texts, str):
                documents_texts_list = [documents_texts]
            else:
                documents_texts_list = documents_texts

            # Qwen3-VL multimodal path: build inputs through processor with
            # text + media together so image tokens are materialized correctly.
            if _is_qwen3_vl_embedding_model():
                from qwen_vl_utils import process_vision_info

                images, videos = process_vision_info([doc])
                # Fallback: process_vision_info may miss in-memory PIL images.
                if not images:
                    inline_images = []
                    for message in doc:
                        for part in message.get("content", []):
                            if part.get("type") == "image" and part.get("image") is not None:
                                inline_images.append(part["image"])
                    if inline_images:
                        images = inline_images

                qwen_kw = {
                    "text": documents_texts_list,
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": True,
                    "max_length": 32768,
                }
                if images is not None:
                    qwen_kw["images"] = images
                if videos is not None:
                    qwen_kw["videos"] = videos
                    qwen_kw["videos_kwargs"] = {
                        "min_pixels": 32 * 14 * 14,
                        "max_pixels": 64 * 28 * 28,
                    }

                batch_dict = processor(**qwen_kw)
            else:
                from qwen_omni_utils import process_mm_info

                audio, images, videos = process_mm_info([doc], use_audio_in_video=False)
                if images is not None:
                    processor_kw["images"] = images
                if videos is not None:
                    processor_kw["videos"] = videos
                    processor_kw["videos_kwargs"] = {
                        "min_pixels": 32 * 14 * 14,
                        "max_pixels": 64 * 28 * 28,
                        "use_audio_in_video": False,
                    }
                if audio is not None:
                    processor_kw["audio"] = audio
                    processor_kw["audio_kwargs"] = {"max_length": 2048000}

                batch_dict = processor(**processor_kw)
        batch_dict = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch_dict.items()
        }

        with torch.no_grad():
            out = model(**batch_dict, output_hidden_states=True)
            hidden = out.hidden_states[-1]
        attention_mask = batch_dict["attention_mask"]
        hidden_masked = hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embedding = hidden_masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embedding = F.normalize(embedding.float(), dim=-1).cpu().numpy()
        return embedding

    except torch.OutOfMemoryError:
        print(f"[Embedding] OOM on {device}! Clearing cache.")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as e:
        print(f"[Embedding] Inference failed on {device}: {e}")
        import traceback

        traceback.print_exc()
        return None
    finally:
        try:
            if "batch_dict" in locals():
                del batch_dict
            if "out" in locals():
                del out
        except Exception:
            pass
        # Aggressive cleanup hurts throughput; keep it opt-in.
        if os.getenv("OMURA_EMBEDDING_AGGRESSIVE_CLEANUP", "false").lower() == "true":
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        # Always return the slot to the pool
        _model_pool.put((model, processor, device))


def _qwen_text_messages(text: str, instruction: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]


def _qwen_image_messages(img: Image.Image, instruction: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "image retrieval query"},
            ],
        },
    ]


def _encode_jina_images(images: list) -> Optional[np.ndarray]:
    """Encode a list of PIL Images with Jina CLIP v2. Returns (n, 1024) float32 array."""
    ensure_model_loaded()
    model, _, device = _model_pool.get()
    try:
        with torch.no_grad():
            embs = model.encode_image(images, normalize=True)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().float().numpy()
        return np.asarray(embs, dtype=np.float32)
    except torch.OutOfMemoryError:
        print(f"[Embedding] OOM on {device} (Jina CLIP image). Clearing cache.")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as exc:
        print(f"[Embedding] Jina CLIP image error: {exc}")
        return None
    finally:
        _model_pool.put((model, _, device))


def _encode_jina_text(texts: list[str]) -> Optional[np.ndarray]:
    """Encode a list of text strings with Jina CLIP v2. Returns (n, 1024) float32 array."""
    ensure_model_loaded()
    model, _, device = _model_pool.get()
    try:
        with torch.no_grad():
            embs = model.encode_text(texts, normalize=True)
        if isinstance(embs, torch.Tensor):
            embs = embs.cpu().float().numpy()
        return np.asarray(embs, dtype=np.float32)
    except torch.OutOfMemoryError:
        print(f"[Embedding] OOM on {device} (Jina CLIP text). Clearing cache.")
        torch.cuda.empty_cache()
        gc.collect()
        return None
    except Exception as exc:
        print(f"[Embedding] Jina CLIP text error: {exc}")
        return None
    finally:
        _model_pool.put((model, _, device))


def _sliding_text_windows(
    text: str, window_chars: int, overlap_chars: int
) -> list[str]:
    if window_chars <= 0:
        return [text]
    overlap_chars = max(0, min(overlap_chars, window_chars - 1))
    step = max(1, window_chars - overlap_chars)
    out: list[str] = []
    n = len(text)
    i = 0
    while i < n:
        chunk = text[i : i + window_chars]
        if chunk:
            out.append(chunk)
        if i + window_chars >= n:
            break
        i += step
    return out


def _encode_omura_emmbed_text(texts: list[str]) -> Optional[np.ndarray]:
    ensure_model_loaded()
    model, processor, device = _model_pool.get()
    try:
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding="max_length",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_text_features"):
                out_or_feats = model.get_text_features(**inputs)
            else:
                out_or_feats = model(**inputs)
            if isinstance(out_or_feats, torch.Tensor):
                feats = out_or_feats
            else:
                feats = getattr(out_or_feats, "text_embeds", None)
                if feats is None:
                    feats = getattr(out_or_feats, "pooler_output", None)
                if (
                    feats is None
                    and isinstance(out_or_feats, tuple)
                    and len(out_or_feats) > 0
                ):
                    feats = out_or_feats[0]
                if feats is None:
                    raise RuntimeError("Omura Embed text output has no embeddable tensor.")
            # Ensure [batch, dim]
            if feats.ndim == 3:
                feats = feats.mean(dim=1)
            elif feats.ndim == 1:
                feats = feats.unsqueeze(0)
            elif feats.ndim > 3:
                feats = feats.reshape(feats.shape[0], -1)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()
    except Exception as e:
        print(f"[Embedding] Omura Embed text error: {e}")
        return None
    finally:
        _model_pool.put((model, processor, device))


def _encode_omura_emmbed_images(images: list[Image.Image]) -> Optional[np.ndarray]:
    ensure_model_loaded()
    model, processor, device = _model_pool.get()
    try:
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if hasattr(model, "get_image_features"):
                out_or_feats = model.get_image_features(**inputs)
            else:
                out_or_feats = model(**inputs)
            if isinstance(out_or_feats, torch.Tensor):
                feats = out_or_feats
            else:
                feats = getattr(out_or_feats, "image_embeds", None)
                if feats is None:
                    feats = getattr(out_or_feats, "pooler_output", None)
                if (
                    feats is None
                    and isinstance(out_or_feats, tuple)
                    and len(out_or_feats) > 0
                ):
                    feats = out_or_feats[0]
                if feats is None:
                    raise RuntimeError("Omura Embed image output has no embeddable tensor.")
            # Ensure [batch, dim]
            if feats.ndim == 3:
                feats = feats.mean(dim=1)
            elif feats.ndim == 1:
                feats = feats.unsqueeze(0)
            elif feats.ndim > 3:
                feats = feats.reshape(feats.shape[0], -1)
        feats = F.normalize(feats.float(), dim=-1)
        return feats.cpu().numpy()
    except Exception as e:
        print(f"[Embedding] Omura Embed image error: {e}")
        return None
    finally:
        _model_pool.put((model, processor, device))


def is_model_ready() -> bool:
    """Return True if at least one model instance is loaded."""
    return _pool_size > 0


def generate_image_embedding(
    image_data: bytes,
    blob_id: str | None = None,
    instruction: str | None = None,
):
    from omura.parsers.multimodal import parse_image

    try:
        img = parse_image(image_data)
        if _is_jina_clip_model():
            max_dim = 512
            if max(img.size) > max_dim:
                scale = max_dim / max(img.size)
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
        elif not _is_omura_emmbed_model():
            max_dim = 1280
            if max(img.size) > max_dim:
                scale = max_dim / max(img.size)
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
        if _is_qwen3_vl_embedding_model() and _USE_FALLBACK_FOR_QWEN:
            vecs = _encode_fallback_images([img])
            return None if vecs is None else vecs[0]
        if _is_jina_clip_model():
            vecs = _encode_jina_images([img])
            return None if vecs is None else vecs[0]
        if _is_omura_emmbed_model():
            vecs = _encode_omura_emmbed_images([img])
            return None if vecs is None else vecs[0]
        qwen_instruction = (instruction or _QWEN_IMAGE_INSTRUCTION).strip()
        messages = _qwen_image_messages(img, qwen_instruction)
        return _encode_documents([messages])[0]
    except _ModelNotReadyError:
        return None
    except Exception as e:
        print(f"[Embedding] Image error ({blob_id}): {e}")
        return None


def generate_video_embedding(video_data: bytes, ext: str, blob_id: str):
    if _is_omura_emmbed_model():
        # Omura Embed is optimized for image+text retrieval in this pipeline.
        return None
    from omura.parsers.multimodal import parse_video

    video_path = None
    try:
        video_path = parse_video(video_data, ext, blob_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "max_pixels": 1280 * 720,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        return _encode_documents([messages])[0]
    except _ModelNotReadyError:
        return None
    except Exception as e:
        print(f"[Embedding] Video error ({blob_id}): {e}")
        return None
    finally:
        if video_path is not None:
            try:
                video_path.unlink(missing_ok=True)
            except Exception:
                pass


def generate_audio_embedding(audio_data: bytes, ext: str, blob_id: str):
    if _is_qwen3_vl_embedding_model():
        # Qwen3-VL embedding model does not support standalone audio inputs.
        return None
    if _is_omura_emmbed_model():
        # Omura Embed is optimized for image+text retrieval in this pipeline.
        return None
    from omura.parsers.multimodal import parse_audio

    audio_path = None
    try:
        audio_path = parse_audio(audio_data, ext, blob_id)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": "Describe this audio."},
                ],
            }
        ]
        return _encode_documents([messages])[0]
    except _ModelNotReadyError:
        return None
    except Exception as e:
        print(f"[Embedding] Audio error ({blob_id}): {e}")
        return None
    finally:
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass


def generate_text_embedding(
    text: str,
    is_document: bool = False,
    instruction: str | None = None,
):
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    # Optional safety fallback: use a stable text+image embedder for Qwen route.
    if _is_qwen3_vl_embedding_model() and _USE_FALLBACK_FOR_QWEN:
        vecs = _encode_fallback_text([cleaned[:8000]])
        return None if vecs is None else vecs[0]

    # Jina CLIP v2: native multilingual, handles both query and document natively
    if _is_jina_clip_model():
        vecs = _encode_jina_text([cleaned[:8000]])
        return None if vecs is None else vecs[0]

    if _is_omura_emmbed_model() and not is_document:
        q = cleaned[:4000]
        vecs = _encode_omura_emmbed_text([q])
        if vecs is None or len(vecs) == 0:
            return None
        return vecs[0]

    if _is_omura_emmbed_model() and is_document:
        window_chars = int(os.getenv("OMURA_TEXT_WINDOW_CHARS", "4000"))
        overlap_chars = int(os.getenv("OMURA_TEXT_WINDOW_OVERLAP_CHARS", "400"))
        max_windows = int(os.getenv("OMURA_TEXT_MAX_WINDOWS", "64"))
        windows = _sliding_text_windows(
            cleaned, window_chars=window_chars, overlap_chars=overlap_chars
        )
        if max_windows > 0 and len(windows) > max_windows:
            windows = windows[:max_windows]
        vecs = _encode_omura_emmbed_text(windows)
        if vecs is None or len(vecs) == 0:
            return None
        avg = np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
        n = np.linalg.norm(avg)
        if n > 0:
            avg = avg / n
        return avg

    if _is_qwen3_vl_embedding_model() and not is_document:
        qwen_instruction = (instruction or _QWEN_QUERY_INSTRUCTION).strip()
        messages = _qwen_text_messages(cleaned[:4000], qwen_instruction)
        try:
            return _encode_documents([messages])[0]
        except _ModelNotReadyError:
            return None
        except Exception as e:
            print(f"[Embedding] Text error: {e}")
            return None

    if not is_document:
        prompt = f"query: {cleaned[:4000]}"
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        try:
            return _encode_documents([messages])[0]
        except _ModelNotReadyError:
            return None
        except Exception as e:
            print(f"[Embedding] Text error: {e}")
            return None

    if _is_qwen3_vl_embedding_model() and is_document:
        window_chars = int(os.getenv("OMURA_TEXT_WINDOW_CHARS", "4000"))
        overlap_chars = int(os.getenv("OMURA_TEXT_WINDOW_OVERLAP_CHARS", "400"))
        max_windows = int(os.getenv("OMURA_TEXT_MAX_WINDOWS", "64"))

        windows = _sliding_text_windows(
            cleaned, window_chars=window_chars, overlap_chars=overlap_chars
        )
        if max_windows > 0 and len(windows) > max_windows:
            windows = windows[:max_windows]

        vecs = []
        try:
            for w in windows:
                qwen_instruction = (instruction or _QWEN_DOCUMENT_INSTRUCTION).strip()
                messages = _qwen_text_messages(w, qwen_instruction)
                emb = _encode_documents([messages])
                if emb is None:
                    continue
                vecs.append(np.asarray(emb[0], dtype=np.float32))
            if not vecs:
                return None
            avg = np.mean(np.stack(vecs, axis=0), axis=0)
            n = np.linalg.norm(avg)
            if n > 0:
                avg = avg / n
            return avg
        except _ModelNotReadyError:
            return None
        except Exception as e:
            print(f"[Embedding] Text error: {e}")
            return None

    # Document mode: use sliding windows to represent the full content.
    window_chars = int(os.getenv("OMURA_TEXT_WINDOW_CHARS", "4000"))
    overlap_chars = int(os.getenv("OMURA_TEXT_WINDOW_OVERLAP_CHARS", "400"))
    max_windows = int(os.getenv("OMURA_TEXT_MAX_WINDOWS", "64"))

    windows = _sliding_text_windows(
        cleaned, window_chars=window_chars, overlap_chars=overlap_chars
    )
    if max_windows > 0 and len(windows) > max_windows:
        windows = windows[:max_windows]

    vecs = []
    try:
        for w in windows:
            prompt = f"passage: {w}"
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            emb = _encode_documents([messages])
            if emb is None:
                continue
            vecs.append(np.asarray(emb[0], dtype=np.float32))
        if not vecs:
            return None
        avg = np.mean(np.stack(vecs, axis=0), axis=0)
        n = np.linalg.norm(avg)
        if n > 0:
            avg = avg / n
        return avg
    except _ModelNotReadyError:
        return None
    except Exception as e:
        print(f"[Embedding] Text error: {e}")
        return None


_nsfw_embedding_cache = None
_nsfw_embeddings_cache = None


def get_nsfw_embeddings():
    """Return cached NSFW prototype embeddings for zero-shot safety scoring."""
    global _nsfw_embeddings_cache
    if _nsfw_embeddings_cache is not None:
        return _nsfw_embeddings_cache

    prompts = [
        "explicit nudity, nude body, sexual content, pornography",
        "adult explicit photo, exposed genitals, explicit sexual act",
        "nsfw erotic content, pornographic image, uncensored nudity",
        "graphic sexual content, explicit adult scene, hentai porn",
        "breasts and genitals visible, explicit nudity close-up",
        "suggestive sexualized content, provocative adult pose, revealing attire",
        "lingerie or bikini suggestive photo, sensual suggestive imagery",
        "implied nudity, borderline nsfw suggestive content",
    ]

    out = []
    try:
        for text in prompts:
            emb = generate_text_embedding(text, is_document=False)
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float32).flatten()
            n = np.linalg.norm(vec)
            if n > 0:
                vec = vec / n
            out.append(vec)
        _nsfw_embeddings_cache = out if out else None
        return _nsfw_embeddings_cache
    except Exception:
        return None


def nsfw_similarity_score_0_100(
    image_vec: np.ndarray,
    nsfw_vecs: Optional[list] = None,
) -> float:
    """Max cosine similarity to NSFW text prototypes, on the same 0–100 scale as ``/search``.

    Uses ``min(max(cosine, 0) * 1000, 100)`` so it matches legacy retrieval scoring.
    """
    vecs = nsfw_vecs if nsfw_vecs is not None else get_nsfw_embeddings()
    if not vecs:
        return 0.0
    v = np.asarray(image_vec, dtype=np.float32).flatten()
    n = float(np.linalg.norm(v))
    if n > 0:
        v = v / n
    best = 0.0
    for proto in vecs:
        p = np.asarray(proto, dtype=np.float32).flatten()
        pn = float(np.linalg.norm(p))
        if pn > 0:
            p = p / pn
        c = float(np.dot(v, p))
        if c > best:
            best = c
    return float(min(max(best, 0.0) * 1000.0, 100.0))


def nsfw_tag_score_min() -> float:
    """Exclusive minimum: tag NSFW when ``nsfw_similarity_score_0_100 >`` this (default 85)."""
    return float(os.getenv("OMURA_NSFW_TAG_SCORE_MIN", "85"))


def is_nsfw_from_tag_score(
    score_0_100: float, *, score_min: Optional[float] = None
) -> bool:
    thr = float(score_min) if score_min is not None else nsfw_tag_score_min()
    return float(score_0_100) > thr


_semantic_category_cache = None


def _semantic_prompts_for_category(category: str) -> list[str]:
    """Return richer prompt variants for robust zero-shot category anchors."""
    c = (category or "").strip().lower()
    prompt_bank = {
        "cat": [
            "a photo of a cat",
            "cat portrait, domestic feline",
            "kitten, house cat, pet cat",
        ],
        "dog": [
            "a photo of a dog",
            "dog portrait, domestic canine",
            "puppy, pet dog, canine animal",
        ],
        "animal": [
            "a photo of an animal",
            "pet or wildlife animal",
            "mammal, bird, reptile, or fish in a natural scene",
        ],
        "wildlife": [
            "wildlife animal in nature",
            "a wild animal outdoors",
            "birds or mammals in natural habitat",
        ],
        "pet": [
            "a pet animal at home",
            "domestic pet cat or dog",
            "household companion animal",
        ],
        "nsfw": [
            "explicit nudity sexual content pornography",
            "adult explicit image, uncensored nudity",
            "graphic sexual content and erotic nudity",
            "suggestive sexualized content, provocative adult pose",
            "revealing attire, sensual suggestive imagery",
        ],
        "pornographic": [
            "pornographic explicit adult content",
            "uncensored explicit sexual act",
            "adult explicit nudity close-up",
            "hardcore explicit porn scene",
            "explicit sexual activity with visible nudity",
        ],
    }
    return prompt_bank.get(c, [f"a photo of a {category}"])


def get_semantic_category_embeddings(categories: list[str]) -> list[np.ndarray]:
    """Return cached or freshly generated embeddings for semantic categories.

    Args:
        categories: A list of category names (e.g. ['cat', 'dog']).

    Returns:
        A list of normalized numpy arrays.
    """
    global _semantic_category_cache

    # If we have a cache, check if it's still valid for the requested categories
    if _semantic_category_cache is not None:
        cached_cats = list(_semantic_category_cache.keys())
        if sorted(cached_cats) == sorted(categories):
            return [_semantic_category_cache[c] for c in categories]

    out = []
    try:
        for cat in categories:
            vecs_for_cat: list[np.ndarray] = []
            for prompt in _semantic_prompts_for_category(cat):
                emb = generate_text_embedding(prompt, is_document=False)
                if emb is None:
                    continue
                vec = np.asarray(emb, dtype=np.float32).flatten()
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / n
                vecs_for_cat.append(vec)
            if not vecs_for_cat:
                continue
            avg = np.mean(np.stack(vecs_for_cat, axis=0), axis=0)
            n = np.linalg.norm(avg)
            if n > 0:
                avg = avg / n
            out.append(avg.astype(np.float32))

        if len(out) == len(categories):
            _semantic_category_cache = dict(zip(categories, out))
            return out
        return []
    except Exception as e:
        print(f"[Embedding] Semantic category error: {e}")
        return []


def generate_multimodal_embedding(text: str, images: list = None):
    """Embed a document with text and optional PIL images (e.g. PDF pages)."""
    if _is_omura_emmbed_model():
        vectors = []
        if images:
            image_vecs = _encode_omura_emmbed_images(images)
            if image_vecs is not None and len(image_vecs) > 0:
                vectors.append(
                    np.mean(np.asarray(image_vecs, dtype=np.float32), axis=0)
                )
        if text:
            text_vec = generate_text_embedding(text[:100000], is_document=True)
            if text_vec is not None:
                vectors.append(np.asarray(text_vec, dtype=np.float32))
        if not vectors:
            return None
        out = np.mean(np.stack(vectors, axis=0), axis=0)
        n = np.linalg.norm(out)
        if n > 0:
            out = out / n
        return out

    content = []
    if images:
        for img in images:
            if max(img.size) > 1024:
                scale = 1024 / max(img.size)
                img = img.resize(
                    (int(img.width * scale), int(img.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            content.append({"type": "image", "image": img})
    if text:
        content.append({"type": "text", "text": text[:30000]})
    elif not images:
        return None
    messages = [{"role": "user", "content": content}]
    try:
        return _encode_documents([messages])[0]
    except _ModelNotReadyError:
        return None
    except Exception as e:
        print(f"[Embedding] Multimodal error: {e}")
        return None
