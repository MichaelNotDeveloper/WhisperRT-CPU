import io
from itertools import islice

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset

TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "normalized_text")
TARGET_SAMPLE_RATE = 16000
PRESETS = {
    "librispeech": ("librispeech_asr", "clean", "test"),
    "golos": ("bond005/sberdevices_golos_10h_crowd", None, "validation"),
    "earnings22": ("distil-whisper/earnings22", "chunked", "test"),
    "earnings-22": ("distil-whisper/earnings22", "chunked", "test"),
}


class AudioTextDataset:
    def __init__(self, name, split=None):
        path, config, default_split = PRESETS.get(name, (name, None, "test"))
        ds = (
            load_dataset(path, config, split=split or default_split, streaming=True)
            if config
            else load_dataset(path, split=split or default_split, streaming=True)
        )
        self.dataset = ds.cast_column("audio", Audio(decode=False))

    def take(self, n):
        return [
            (self._load_audio(r["audio"]), self._text(r)) for r in islice(self.dataset, max(n, 0))
        ]

    def _load_audio(self, a):
        x, sr = (
            sf.read(io.BytesIO(a["bytes"]), dtype="float32")
            if a.get("bytes") is not None
            else sf.read(a["path"], dtype="float32")
        )
        if x.ndim == 2:
            x = x.mean(axis=1)
        x = np.asarray(x, dtype=np.float32)
        if int(sr) != TARGET_SAMPLE_RATE:
            x = torchaudio.functional.resample(
                torch.from_numpy(x),
                int(sr),
                TARGET_SAMPLE_RATE,
            ).numpy()
            sr = TARGET_SAMPLE_RATE
        return {"array": x, "sampling_rate": int(sr)}

    def _text(self, row):
        for c in TEXT_COLUMNS:
            if c in row:
                return row[c]
        raise KeyError(f"text column not found: {list(row)}")
