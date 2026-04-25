from itertools import islice
import io
import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio

TEXT_COLUMNS = ("text", "sentence", "transcription", "transcript", "normalized_text")
PRESETS = {
    "librispeech": ("librispeech_asr", "clean", "test"),
    "golos": ("bond005/sberdevices_golos_10h_crowd", None, "validation"),
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
        return {"array": np.asarray(x, dtype=np.float32), "sampling_rate": int(sr)}

    def _text(self, row):
        for c in TEXT_COLUMNS:
            if c in row:
                return row[c]
        raise KeyError(f"text column not found: {list(row)}")
