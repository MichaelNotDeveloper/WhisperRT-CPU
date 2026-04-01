import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from faster_whisper import WhisperModel
from jiwer import Compose, ExpandCommonEnglishContractions, RemoveMultipleSpaces, RemovePunctuation, Strip, ToLowerCase, wer
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = ROOT / "faster_whisper_quant_artifacts"
DATA_DIR = ROOT / "data" / "librispeech"
SUBSET_MANIFEST = ROOT / "data" / "librispeech_subset" / "manifest.jsonl"
SAMPLE_RATE = 16000
MAX_FILES = int(os.environ.get("WHISPER_MAX_FILES", "20"))
CPU_THREADS = int(os.environ.get("WHISPER_CPU_THREADS", "16"))
MODEL_SIZE = os.environ.get("FASTER_WHISPER_MODEL_SIZE", "large-v3")
ONLY_EXPERIMENTS = {
    name.strip() for name in os.environ.get("WHISPER_ONLY_EXPERIMENTS", "").split(";") if name.strip()
}

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(CPU_THREADS)


@dataclass
class Example:
    audio: torch.Tensor
    sample_rate: int
    reference: str


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with (ARTIFACTS_DIR / "run.log").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def compute_wer(predictions: list[str], references: list[str]) -> float:
    transform = Compose(
        [
            RemovePunctuation(),
            ExpandCommonEnglishContractions(),
            ToLowerCase(),
            RemoveMultipleSpaces(),
            Strip(),
        ]
    )
    return wer(transform(references), transform(predictions))


def compute_rtf(inference_time_sec: float, audio_len_sec: float) -> float:
    return inference_time_sec / audio_len_sec


def load_librispeech_subset(max_files: int = MAX_FILES) -> list[Example]:
    examples: list[Example] = []
    if SUBSET_MANIFEST.exists():
        for line in SUBSET_MANIFEST.read_text(encoding="utf-8").splitlines()[:max_files]:
            item = json.loads(line)
            audio, sr = sf.read(SUBSET_MANIFEST.parent / item["audio_path"])
            waveform = torch.as_tensor(audio, dtype=torch.float32)
            if waveform.ndim > 1:
                waveform = waveform.mean(dim=1)
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
                sr = SAMPLE_RATE
            examples.append(Example(audio=waveform.contiguous(), sample_rate=sr, reference=item["text"]))
        return examples

    dataset = load_dataset("librispeech_asr", "clean", split=f"test[:{max_files}]")
    for item in dataset:
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=1)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, SAMPLE_RATE).squeeze(0)
            sr = SAMPLE_RATE
        examples.append(Example(audio=waveform.contiguous(), sample_rate=sr, reference=item["text"]))
    return examples


def run_experiment(
    examples: list[Example],
    experiment_name: str,
    device: str,
    compute_type: str,
    beam_size: int = 1,
    warmup_examples: int = 1,
) -> dict[str, Any]:
    model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type, cpu_threads=CPU_THREADS)
    predictions: list[str] = []
    references: list[str] = []
    total_inference_time = 0.0
    total_audio_time = 0.0

    for idx, ex in tqdm(list(enumerate(examples)), total=len(examples), desc=experiment_name):
        start = time.perf_counter()
        segments, _ = model.transcribe(ex.audio.numpy(), beam_size=beam_size, language="en")
        transcription = "".join(segment.text for segment in segments)
        end = time.perf_counter()

        if idx < warmup_examples:
            continue

        total_inference_time += end - start
        total_audio_time += ex.audio.shape[-1] / ex.sample_rate
        predictions.append(transcription)
        references.append(ex.reference)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": experiment_name,
        "status": "ok",
        "error": None,
        "wer": compute_wer(predictions, references),
        "rtf": compute_rtf(total_inference_time, total_audio_time),
        "total_inference_time": total_inference_time,
        "total_audio_time": total_audio_time,
    }


def load_existing_rows() -> list[dict[str, Any]]:
    path = ARTIFACTS_DIR / "results.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def save_row(row: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    by_name = {r["name"]: r for r in rows}
    by_name[row["name"]] = row
    ordered = [by_name[k] for k in sorted(by_name)]
    pd.DataFrame(ordered).to_csv(ARTIFACTS_DIR / "results.csv", index=False)
    with (ARTIFACTS_DIR / f"{row['name'].replace('|', '__').replace('/', '_')}.json").open("w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)


def main() -> None:
    log("Preparing dataset")
    examples = load_librispeech_subset(MAX_FILES)
    log(f"Loaded {len(examples)} examples")
    experiments = [
        ("faster-whisper|CPU|int8", "cpu", "int8"),
        ("faster-whisper|CPU|int8_float32", "cpu", "int8_float32"),
        ("faster-whisper|GPU|int8_float16", "cuda", "int8_float16"),
        ("faster-whisper|GPU|int8", "cuda", "int8"),
    ]
    if ONLY_EXPERIMENTS:
        experiments = [exp for exp in experiments if exp[0] in ONLY_EXPERIMENTS]

    rows = load_existing_rows()
    done = {r["name"] for r in rows if r.get("status") == "ok"}

    for name, device, compute_type in experiments:
        if name in done:
            log(f"Skipping already completed {name}")
            continue

        start = time.perf_counter()
        log(f"Starting {name}")
        try:
            row = run_experiment(examples, name, device, compute_type)
            row["wall_time_sec"] = time.perf_counter() - start
        except Exception as exc:
            row = {
                "name": name,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "wer": None,
                "rtf": None,
                "total_inference_time": None,
                "total_audio_time": None,
                "wall_time_sec": time.perf_counter() - start,
            }
            log(f"Failed {name}: {row['error']}")

        rows = [r for r in rows if r["name"] != name] + [row]
        save_row(row, rows)
        log(f"Finished {name} in {row['wall_time_sec'] / 60:.2f} min")

    df = pd.read_csv(ARTIFACTS_DIR / "results.csv").sort_values("name")
    log("Final results:")
    log(df.to_string(index=False))


if __name__ == "__main__":
    main()
