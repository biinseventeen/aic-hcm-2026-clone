"""The text encoder for the dense retrieval channel.

A non-negotiable constraint
---------------------------
The text encoder must live in the **same embedding space** as the indexed image features. The
current index uses the organiser-supplied features from ``clip-ViT-B-32``, so the text tower
*must* be the text tower of that exact checkpoint. Using a different model — even a stronger
one such as SigLIP — produces meaningless cosine similarities, and the failure is **silent**:
scores still fall in [-1, 1], ranks still exist, and the result still looks like a valid list.
That is why :func:`load_text_encoder` checks the dimensionality and refuses to run on a
mismatch.

Three loading paths, tried in order
-----------------------------------
1. ``sentence-transformers`` — ``SentenceTransformer("clip-ViT-B-32")``. This is the exact name
   the rules give, so it is the closest match.
2. ``open_clip`` — ``ViT-B-32`` with the ``openai`` pretrained weights.
3. ``transformers`` — ``CLIPModel.from_pretrained("openai/clip-vit-base-patch32")``.

If no package is available, :class:`StubTextEncoder` is used only when explicitly requested via
``allow_stub=True``, and it must **never** be used to produce a submission.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..log import log

__all__ = ["EncoderUnavailable", "StubTextEncoder", "TextEncoder", "is_stub", "load_text_encoder"]


class EncoderUnavailable(RuntimeError):
    """No compatible text encoder could be loaded."""


@runtime_checkable
class TextEncoder(Protocol):
    """The minimal interface the retrieval layer needs."""

    name: str
    dim: int

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return (len(texts), dim) fp32, L2-normalised."""
        ...


def _l2_normalise(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


@dataclass
class SentenceTransformerEncoder:
    name: str = "clip-ViT-B-32"
    dim: int = 512
    _model: object = None

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self._model = SentenceTransformer(self.name)
        self.dim = int(self._model.get_sentence_embedding_dimension())  # type: ignore[union-attr]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(  # type: ignore[union-attr]
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        return _l2_normalise(vectors)


@dataclass
class OpenClipEncoder:
    name: str = "ViT-B-32"
    pretrained: str = "openai"
    dim: int = 512
    device: str = "cpu"
    _model: object = None
    _tokenizer: object = None

    def __post_init__(self) -> None:
        import open_clip
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, _ = open_clip.create_model_and_transforms(self.name, pretrained=self.pretrained)
        model.eval().to(self.device)
        self._model = model
        self._tokenizer = open_clip.get_tokenizer(self.name)
        with torch.no_grad():
            probe = self._tokenizer(["x"]).to(self.device)  # type: ignore[operator]
            self.dim = int(model.encode_text(probe).shape[1])

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            tokens = self._tokenizer(list(texts)).to(self.device)  # type: ignore[operator]
            vectors = self._model.encode_text(tokens)  # type: ignore[union-attr]
        return _l2_normalise(vectors.float().cpu().numpy())


@dataclass
class HuggingFaceClipEncoder:
    name: str = "openai/clip-vit-base-patch32"
    dim: int = 512
    device: str = "cpu"
    _model: object = None
    _tokenizer: object = None

    def __post_init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPTokenizerFast

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(self.name).eval().to(self.device)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(self.name)
        self.dim = int(self._model.config.projection_dim)  # type: ignore[union-attr]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        batch = self._tokenizer(  # type: ignore[operator]
            list(texts), padding=True, truncation=True, return_tensors="pt"
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.no_grad():
            vectors = self._model.get_text_features(**batch)  # type: ignore[union-attr]
        return _l2_normalise(vectors.float().cpu().numpy())


@dataclass
class StubTextEncoder:
    """A fake encoder, **for pipeline testing only**. It carries no semantics whatsoever.

    Produces deterministic vectors from the SHA-256 hash of the string. Use it to check that
    data flows correctly through RRF, the task layer and the submission writer on a machine
    with no model installed.

    It is **not** in the same space as the image features, so every similarity it produces is
    noise. Any function that writes a submission must refuse an encoder with ``is_stub = True``.
    """

    name: str = "stub-sha256"
    dim: int = 512
    is_stub: bool = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            # Expand the 32-byte digest to `dim` numbers by iterated hashing — deterministic.
            buffer = bytearray()
            seed = digest
            while len(buffer) < self.dim * 4:
                seed = hashlib.sha256(seed).digest()
                buffer.extend(seed)
            raw = np.frombuffer(bytes(buffer[: self.dim * 4]), dtype=np.uint32)
            out[i] = raw.astype(np.float32) / np.float32(2**31) - 1.0
        return _l2_normalise(out)


def load_text_encoder(
    model: str = "clip-ViT-B-32",
    *,
    expect_dim: int | None = None,
    allow_stub: bool = False,
    verbose: bool = True,
) -> TextEncoder:
    """Load a text encoder and check its dimensionality against the index.

    ``expect_dim`` should always be passed, taken from ``DenseIndex.dim``. A dimension mismatch
    is a *blocking* error, not a warning.
    """
    errors: list[str] = []
    for factory, label in (
        (lambda: SentenceTransformerEncoder(model), "sentence-transformers"),
        (OpenClipEncoder, "open_clip"),
        (HuggingFaceClipEncoder, "transformers"),
    ):
        try:
            encoder = factory()
        except Exception as exc:  # ImportError, download failure, out of VRAM, ...
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue
        if expect_dim is not None and encoder.dim != expect_dim:
            errors.append(
                f"{label}: dimension {encoder.dim} != index dimension {expect_dim} — "
                "not the same embedding space, refusing to use it"
            )
            continue
        if verbose:
            log.info("  text encoder: %s / %s, %d dimensions", label, encoder.name, encoder.dim)
        return encoder  # type: ignore[return-value]

    if allow_stub:
        log.warning(
            "  [!] USING THE STUB ENCODER. Retrieval results are NOISE and exist only to "
            "exercise the pipeline. Never build a submission from this configuration.\n"
            "      Reasons the real models could not be loaded:\n"
            + "\n".join(f"        - {error}" for error in errors)
        )
        return StubTextEncoder(dim=expect_dim or 512)  # type: ignore[return-value]

    raise EncoderUnavailable(
        "could not load a text encoder compatible with the index.\n"
        + "\n".join(f"  - {error}" for error in errors)
        + "\n\nInstall an encoder into the workspace environment:\n"
        "  uv sync --extra encoder      # sentence-transformers + torch\n"
        "  uv add open_clip_torch       # or either of these alternatives\n"
        "  uv add transformers\n"
        "Or pass allow_stub=True to exercise the pipeline (NOT for submissions)."
    )


def is_stub(encoder: object) -> bool:
    return bool(getattr(encoder, "is_stub", False))
