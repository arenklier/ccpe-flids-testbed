"""Update compression schemes and a wire codec for model parameters.

A codec turns an ordered list of float32 tensors (the model delta or the model
itself) into raw bytes and back, so the server and clients agree byte-for-byte
and we can measure communication cost exactly. Three schemes:

  none  : flat float32 concatenation (baseline, lossless)
  int8  : per-tensor symmetric linear quantization to int8 + float32 scale
  topk  : keep the top-k% magnitude entries as (index, value) pairs, rest zero

Shapes are known from the reference model on both ends, so only payloads travel.
"""
from __future__ import annotations

import struct
from typing import List

import numpy as np

# ---- helpers ---------------------------------------------------------------

def _flatten(tensors: List[np.ndarray]) -> tuple[np.ndarray, list[tuple]]:
    shapes = [t.shape for t in tensors]
    flat = np.concatenate([t.ravel().astype(np.float32) for t in tensors])
    return flat, shapes


def _unflatten(flat: np.ndarray, shapes: list[tuple]) -> List[np.ndarray]:
    out, off = [], 0
    for shp in shapes:
        n = int(np.prod(shp)) if shp else 1
        out.append(flat[off:off + n].reshape(shp).astype(np.float32))
        off += n
    return out


# ---- codecs ----------------------------------------------------------------

def encode(tensors: List[np.ndarray], scheme: str, topk_frac: float = 0.10) -> bytes:
    flat, _ = _flatten(tensors)
    if scheme == "none":
        return b"N" + flat.tobytes()

    if scheme == "int8":
        amax = float(np.abs(flat).max()) or 1e-8
        scale = amax / 127.0
        q = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
        return b"Q" + struct.pack("<f", scale) + q.tobytes()

    if scheme == "topk":
        n = flat.size
        k = max(1, int(n * topk_frac))
        idx = np.argpartition(np.abs(flat), n - k)[n - k:].astype(np.int32)
        vals = flat[idx].astype(np.float32)
        return (b"T" + struct.pack("<II", n, k)
                + idx.tobytes() + vals.tobytes())

    raise ValueError(f"unknown scheme {scheme!r}")


def decode(blob: bytes, shapes: list[tuple]) -> List[np.ndarray]:
    tag, body = blob[:1], blob[1:]
    if tag == b"N":
        flat = np.frombuffer(body, dtype=np.float32).copy()
    elif tag == b"Q":
        scale = struct.unpack("<f", body[:4])[0]
        q = np.frombuffer(body[4:], dtype=np.int8).astype(np.float32)
        flat = q * scale
    elif tag == b"T":
        n, k = struct.unpack("<II", body[:8])
        idx = np.frombuffer(body[8:8 + 4 * k], dtype=np.int32)
        vals = np.frombuffer(body[8 + 4 * k:], dtype=np.float32)
        flat = np.zeros(n, dtype=np.float32)
        flat[idx] = vals
    else:
        raise ValueError(f"unknown codec tag {tag!r}")
    return _unflatten(flat, shapes)
