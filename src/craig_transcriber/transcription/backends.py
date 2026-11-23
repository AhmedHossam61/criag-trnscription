"""
Stage 2: ASR backends.

Each backend loads lazily and returns (text, language_tag) per chunk.
Only one is ever resident, which is what keeps this inside 8 GB.

`language` is one of:
  "ar"   -- force the Arabic tag. Correct when Arabic is the matrix language
            and English appears as embedded words or short phrases.
  "en"   -- force the English tag.
  "auto" -- decide per chunk. Uses the model's own language ID where it has
            one; falls back to a dual-tag scoring pass where it doesn't.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from .audio import write_wav
from .language import label_script
from .settings import SR


class Backend:
    name = "base"
    default_batch = 1             # max chunks per batch
    default_batch_seconds = 60.0  # max padded audio per batch (count x longest)
    supports_hotwords = False

    def __init__(self, language: str = "auto", hotwords: str | None = None):
        self.language = language
        self.hotwords = hotwords
        self.beams = 1  # beam search width; backends that support it read this in load()

    def load(self) -> None:
        raise NotImplementedError

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        """
        Return one (text, language_tag) pair per input.

        `audio` is float32 16 kHz mono in [-1, 1]; `max_new_tokens` is sized
        to the longest chunk in the batch.
        """
        raise NotImplementedError

    def unload(self) -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


class QwenCleoBackend(Backend):
    """
    mohammedaly22/QwenCleo-ASR -- Qwen3-ASR-1.7B fine-tuned on Egyptian
    podcast speech AND Arabic/English code-switching. The only model here
    trained end-to-end for mixed speech: it keeps embedded English in Latin
    script (`React`, `already`, `soft skills`) instead of transliterating it
    into Arabic letters. Best choice for a bilingual Discord call.

        pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
        pip install qwencleo-asr --no-deps
        pip install "qwen-asr>=0.0.6" numpy soundfile huggingface_hub
    """
    name = "qwencleo"
    default_batch = 16
    default_batch_seconds = 120.0

    _TAGS = {"ar": "Arabic", "en": "English", "auto": None}

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel
        from qwencleo_asr import DEFAULT_MODEL_ID
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
        # Load the underlying Qwen3-ASR model directly rather than through the
        # QwenCleoASR wrapper: the wrapper's transcribe() loops over its inputs
        # one at a time, so it never actually batches. Same checkpoint, dtype
        # and device the wrapper would use.
        self.model = Qwen3ASRModel.from_pretrained(
            DEFAULT_MODEL_ID,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=-1,  # batches are sized by make_batches()
        )
        if self.beams > 1:
            # qwen_asr calls generate() with max_new_tokens only; extra
            # generation options go straight through to transformers.
            import functools
            hf_model = self.model.model
            hf_model.generate = functools.partial(hf_model.generate, num_beams=self.beams)

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        self.model.max_new_tokens = max_new_tokens
        results = self.model.transcribe(
            [(a, SR) for a in audio], language=self._TAGS[self.language]
        )

        out: list[tuple[str, str]] = []
        for r in results:
            text = getattr(r, "text", str(r))
            # The model emits its own language tag; fall back to script.
            detected = getattr(r, "language", None)
            out.append((text, (detected or label_script(text) or self.language)))
        return out


class Qwen3ASRBackend(Backend):
    """
    Qwen/Qwen3-ASR-1.7B-hf -- QwenCleo's base. Genuinely multilingual with
    built-in language identification, so `--language auto` is reliable here.
    Weaker on Egyptian dialect than the fine-tune, but it is the backend to
    reach for if QwenCleo produces something odd, and it is the only one
    that accepts a hotword glossary.

        pip install "transformers>=5.13.0" torch soundfile librosa accelerate
    """
    name = "qwen3asr"
    default_batch = 16
    default_batch_seconds = 120.0
    supports_hotwords = True

    _TAGS = {"ar": "Arabic", "en": "English", "auto": None}

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, AutoModelForMultimodalLM
        model_id = "Qwen/Qwen3-ASR-1.7B-hf"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        ).eval()

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        import torch
        tag = self._TAGS[self.language]

        # apply_transcription_request is fed file paths, the input form its
        # docs show; the files only live for this batch.
        with tempfile.TemporaryDirectory(prefix="craig_batch_") as tmp:
            paths = [str(write_wav(a, Path(tmp) / f"{i}.wav")) for i, a in enumerate(audio)]
            request = {"audio": paths}
            if tag is not None:
                request["language"] = [tag] * len(paths)
            if self.hotwords:
                # Biases decoding toward your jargon so it stays in Latin script.
                request["prompt"] = self.hotwords

            inputs = self.processor.apply_transcription_request(**request).to(
                self.model.device, self.model.dtype
            )
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        generated = output_ids[:, inputs["input_ids"].shape[1]:]

        parsed = self.processor.decode(generated, return_format="parsed")
        out: list[tuple[str, str]] = []
        for item in parsed:
            text = item.get("transcription", "")
            lang = (item.get("language") or "").lower()[:2] or label_script(text)
            out.append((text, lang))
        return out


class CohereArabicBackend(Backend):
    """
    CohereLabs/cohere-transcribe-arabic-07-2026 -- 2B FastConformer encoder,
    lowest WER of any open model on the Open Universal Arabic ASR
    Leaderboard (25.87 avg) and by far the fastest option here.

    Caveat that matters for a bilingual call: the model card states it has no
    automatic language detection and shows inconsistent performance on
    code-switched audio. With `--language auto` this backend therefore runs
    each chunk under BOTH tags and keeps whichever the decoder scored higher.
    That doubles its cost -- still fast, but it is a workaround for a real
    limitation, not a feature. Prefer QwenCleo if English shows up often.

    Gated: accept the terms on the model page, then `hf auth login`.

        pip install "transformers>=5.4.0" torch soundfile librosa sentencepiece protobuf accelerate
    """
    name = "cohere"
    default_batch = 16
    default_batch_seconds = 120.0

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        model_id = "CohereLabs/cohere-transcribe-arabic-07-2026"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.float16, device_map="auto"
        ).eval()

    def _decode_with_score(self, audio, tag: str, max_new_tokens: int) -> list[tuple[str, float]]:
        """Transcribe under one language tag, returning mean token log-prob."""
        import torch

        inputs = self.processor(audio, sampling_rate=SR, return_tensors="pt", language=tag)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)

        with torch.inference_mode():
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded = self.processor.decode(result.sequences, skip_special_tokens=True)
        texts = [decoded] if isinstance(decoded, str) else list(decoded)

        try:
            transition = self.model.compute_transition_scores(
                result.sequences, result.scores, normalize_logits=True
            )
            finite = torch.isfinite(transition)
            totals = (transition * finite).sum(dim=1)
            counts = finite.sum(dim=1).clamp(min=1)
            scores = (totals / counts).tolist()
        except Exception:  # noqa: BLE001 -- scoring is best-effort
            scores = [float("-inf")] * len(texts)

        return list(zip(texts, scores))

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        if self.language in ("ar", "en"):
            return [
                (text, self.language)
                for text, _ in self._decode_with_score(audio, self.language, max_new_tokens)
            ]

        # auto: score both tags and keep the more confident transcription.
        arabic = self._decode_with_score(audio, "ar", max_new_tokens)
        english = self._decode_with_score(audio, "en", max_new_tokens)

        out: list[tuple[str, str]] = []
        for (ar_text, ar_score), (en_text, en_score) in zip(arabic, english):
            if en_score > ar_score:
                out.append((en_text, "en"))
            else:
                out.append((ar_text, "ar"))
        return out


class FasterWhisperBackend(Backend):
    """
    Baseline only. Whisper large-v3 flattens Egyptian into MSA and is the
    worst of these on code-switching -- it tends to transliterate English
    into Arabic script rather than keeping it in Latin. Useful for
    comparison, not production.

        pip install faster-whisper
    """
    name = "faster-whisper"
    default_batch = 1

    def __init__(self, language: str = "auto", hotwords: str | None = None,
                 model_size: str = "large-v3"):
        super().__init__(language, hotwords)
        self.model_size = model_size

    def load(self) -> None:
        from faster_whisper import WhisperModel
        # int8_float16 keeps large-v3 near 3 GB, leaving headroom on an 8 GB card.
        self.model = WhisperModel(self.model_size, device="cuda", compute_type="int8_float16")

    def transcribe_batch(self, audio: list[np.ndarray], max_new_tokens: int) -> list[tuple[str, str]]:
        # Whisper decodes one input at a time and has its own length limits,
        # so max_new_tokens is not applied here.
        out: list[tuple[str, str]] = []
        for samples in audio:
            segments, info = self.model.transcribe(
                samples,
                language=None if self.language == "auto" else self.language,
                beam_size=5,
                vad_filter=False,          # our zero-run gate already did this
                condition_on_previous_text=False,  # stops hallucination cascades
                initial_prompt=self.hotwords or None,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            out.append((text, getattr(info, "language", "") or label_script(text)))
        return out


BACKENDS: dict[str, type[Backend]] = {
    "qwencleo": QwenCleoBackend,
    "qwen3asr": Qwen3ASRBackend,
    "cohere": CohereArabicBackend,
    "faster-whisper": FasterWhisperBackend,
}
