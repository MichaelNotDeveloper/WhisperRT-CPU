from __future__ import annotations

"""
A100-oriented distillation pipeline for:
  teacher: openai/whisper-large-v3-turbo (4 decoder layers)
  student: keep decoder layers [0, 3] -> 2 decoder layers

The dataset presets below are sourced from official Whisper / Hugging Face docs:
  1. Fine-tune Whisper blog (Common Voice 11 English)
     https://huggingface.co/blog/fine-tune-whisper
  2. Whisper large-v3-turbo model card (LibriSpeech / librispeech_long examples)
     https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/README.md

Usage example:
  python3 traning/distill_large_v3_turbo_a100.py \
      --epochs 2 \
      --train-presets official-librispeech-clean official-common-voice-en \
      --max-train-samples 16000 \
      --max-eval-samples 256 \
      --max-latency-samples 64
"""

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOCAL_CACHE_HOME = Path(__file__).resolve().parent / ".cache"
LOCAL_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(LOCAL_CACHE_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_HOME / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

if not os.environ.get("DISPLAY"):
    import matplotlib

    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, Dataset, concatenate_datasets, load_dataset
from jiwer import cer, wer
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
)
from transformers.models.whisper.modeling_whisper import shift_tokens_right

SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class DatasetPreset:
    key: str
    path: str
    config: str | None
    train_split: str | None
    eval_split: str | None
    text_column: str
    source_url: str
    note: str = ""
    auth_required: bool = False


OFFICIAL_DATASETS = {
    "official-librispeech-clean": DatasetPreset(
        key="official-librispeech-clean",
        path="librispeech_asr",
        config="clean",
        train_split="train.100",
        eval_split="test.clean",
        text_column="text",
        source_url="https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/README.md",
        note="Open dataset used throughout Whisper model-card examples.",
    ),
    "official-common-voice-en": DatasetPreset(
        key="official-common-voice-en",
        path="mozilla-foundation/common_voice_11_0",
        config="en",
        train_split="train+validation",
        eval_split="test",
        text_column="sentence",
        source_url="https://huggingface.co/blog/fine-tune-whisper",
        note="Exact dataset family used in the official Whisper fine-tuning guide.",
        auth_required=True,
    ),
    "official-librispeech-long": DatasetPreset(
        key="official-librispeech-long",
        path="distil-whisper/librispeech_long",
        config="clean",
        train_split=None,
        eval_split="validation",
        text_column="text",
        source_url="https://huggingface.co/openai/whisper-large-v3-turbo/blob/main/README.md",
        note="Long-form set used in the turbo model card for latency-oriented examples.",
    ),
}


def parse_args() -> argparse.Namespace:
    # Собирает CLI-интерфейс со всеми основными параметрами обучения, оценки и рантайма.
    parser = argparse.ArgumentParser(
        description="Distill whisper-large-v3-turbo down to 2 decoder layers on A100.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--teacher-model",
        default="openai/whisper-large-v3-turbo",
        help="Teacher checkpoint. Must stay the 4-layer turbo baseline for this setup.",
    )
    model_group.add_argument(
        "--student-layers",
        type=int,
        nargs="+",
        default=[0, 3],
        help="Teacher decoder layer indices to keep in the student.",
    )
    model_group.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze the student encoder. Kept enabled to match the requested recipe.",
    )
    model_group.add_argument(
        "--train-proj-out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the LM head / proj_out together with the decoder.",
    )
    model_group.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the student decoder.",
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument(
        "--train-presets",
        nargs="+",
        default=["official-librispeech-clean", "official-common-voice-en"],
        choices=sorted(OFFICIAL_DATASETS),
        help="Official dataset presets to concatenate for training.",
    )
    data_group.add_argument(
        "--eval-preset",
        default="official-librispeech-clean",
        choices=sorted(OFFICIAL_DATASETS),
        help="Preset used for WER/CER evaluation.",
    )
    data_group.add_argument(
        "--latency-preset",
        default="official-librispeech-long",
        choices=sorted(OFFICIAL_DATASETS),
        help="Preset used for latency benchmarking.",
    )
    data_group.add_argument("--language", default="english")
    data_group.add_argument("--task", default="transcribe")
    data_group.add_argument("--max-train-samples", type=int, default=None)
    data_group.add_argument("--max-eval-samples", type=int, default=256)
    data_group.add_argument("--max-latency-samples", type=int, default=64)
    data_group.add_argument(
        "--dataset-cache-dir",
        default=None,
        help="Optional datasets cache dir.",
    )
    data_group.add_argument(
        "--max-label-tokens",
        type=int,
        default=256,
        help="Tokenizer truncation limit for target text.",
    )

    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--epochs", type=int, default=2)
    optim_group.add_argument("--train-batch-size", type=int, default=8)
    optim_group.add_argument("--eval-batch-size", type=int, default=8)
    optim_group.add_argument("--latency-batch-size", type=int, default=1)
    optim_group.add_argument("--num-workers", type=int, default=4)
    optim_group.add_argument("--learning-rate", type=float, default=1e-4)
    optim_group.add_argument("--weight-decay", type=float, default=0.01)
    optim_group.add_argument("--warmup-ratio", type=float, default=0.05)
    optim_group.add_argument("--gradient-accumulation-steps", type=int, default=4)
    optim_group.add_argument("--max-grad-norm", type=float, default=1.0)
    optim_group.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="A100-friendly mixed precision mode. bf16 is the recommended default.",
    )

    loss_group = parser.add_argument_group("distillation-loss")
    loss_group.add_argument("--ce-weight", type=float, default=0.5)
    loss_group.add_argument("--kl-weight", type=float, default=0.5)
    loss_group.add_argument("--kd-temperature", type=float, default=2.0)

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--seed", type=int, default=42)
    runtime_group.add_argument("--device", default="cuda")
    runtime_group.add_argument(
        "--output-dir",
        default="traning/runs/distill_large_v3_turbo_2layer",
    )
    runtime_group.add_argument("--log-interval", type=int, default=10)
    runtime_group.add_argument("--eval-interval", type=int, default=100)
    runtime_group.add_argument("--save-interval", type=int, default=0)
    runtime_group.add_argument("--plot-interval", type=int, default=1)
    runtime_group.add_argument(
        "--disable-live-plots",
        action="store_true",
        help="Disable matplotlib live plot refreshes and keep only JSONL logging.",
    )

    infer_group = parser.add_argument_group("generation")
    infer_group.add_argument("--generation-max-new-tokens", type=int, default=128)
    infer_group.add_argument("--generation-num-beams", type=int, default=1)
    infer_group.add_argument(
        "--condition-on-prev-tokens",
        action="store_true",
        help="Enable Whisper long-form conditioning. Disabled by default for speed.",
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    # Фиксирует сиды и быстрые CUDA-настройки для воспроизводимости и стабильного throughput.
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def normalise_text(text: str) -> str:
    # Приводит текст к простому каноническому виду для обучения и расчета метрик.
    return " ".join(str(text).strip().lower().split())


def serialize_args(args: argparse.Namespace) -> dict[str, Any]:
    # Преобразует аргументы запуска в JSON-совместимый словарь и добавляет источники датасетов.
    out = vars(args).copy()
    out["official_dataset_sources"] = {
        key: {"path": preset.path, "config": preset.config, "source_url": preset.source_url}
        for key, preset in OFFICIAL_DATASETS.items()
    }
    return out


def dtype_from_name(name: str) -> torch.dtype:
    # Маппит строковое имя precision-режима в torch dtype.
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def autocast_context(device: torch.device, amp_dtype: torch.dtype):
    # Возвращает корректный autocast-контекст для mixed precision на GPU или no-op на CPU/FP32.
    if device.type != "cuda" or amp_dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def count_parameters(model: nn.Module, *, trainable_only: bool = False) -> int:
    # Считает общее или только обучаемое число параметров модели.
    params = (
        parameter
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
    return sum(parameter.numel() for parameter in params)


def load_split(preset: DatasetPreset, split: str, cache_dir: str | None) -> Dataset:
    # Загружает один split датасета, приводит аудио к 16 кГц и оставляет только audio/text.
    load_kwargs = {"split": split}
    if cache_dir is not None:
        load_kwargs["cache_dir"] = cache_dir

    if preset.config is None:
        dataset = load_dataset(preset.path, **load_kwargs)
    else:
        dataset = load_dataset(preset.path, preset.config, **load_kwargs)

    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    if preset.text_column != "text":
        dataset = dataset.rename_column(preset.text_column, "text")

    keep_columns = {"audio", "text"}
    remove_columns = [column for column in dataset.column_names if column not in keep_columns]
    if remove_columns:
        dataset = dataset.remove_columns(remove_columns)
    return dataset


def maybe_subset(dataset: Dataset, max_samples: int | None, seed: int, shuffle: bool) -> Dataset:
    # По необходимости перемешивает датасет и берет ограниченное число примеров.
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if max_samples is None:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_training_dataset(args: argparse.Namespace) -> tuple[Dataset, list[str]]:
    # Загружает и склеивает все выбранные official train preset-ы в единый train dataset.
    loaded = []
    source_messages = []
    for preset_name in args.train_presets:
        preset = OFFICIAL_DATASETS[preset_name]
        if preset.train_split is None:
            raise ValueError(f"{preset_name} does not define a train split")
        try:
            dataset = load_split(preset, preset.train_split, args.dataset_cache_dir)
            loaded.append(dataset)
            source_messages.append(
                f"{preset_name}: {preset.path}/{preset.config or '-'} "
                f"({preset.train_split}) <- {preset.source_url}"
            )
        except Exception as exc:
            tqdm.write(
                f"[dataset warning] skipped {preset_name}: {exc}"
                f"{' (auth/terms likely required)' if preset.auth_required else ''}"
            )

    if not loaded:
        raise RuntimeError("No train datasets could be loaded from the official presets.")

    train_dataset = loaded[0] if len(loaded) == 1 else concatenate_datasets(loaded)
    train_dataset = maybe_subset(
        train_dataset,
        max_samples=args.max_train_samples,
        seed=args.seed,
        shuffle=True,
    )
    return train_dataset, source_messages


def load_eval_dataset(
    preset_name: str,
    args: argparse.Namespace,
    max_samples: int | None,
    *,
    shuffle: bool,
) -> tuple[Dataset, str]:
    # Загружает один датасет для валидации или latency-бенчмарка и возвращает строку-описание источника.
    preset = OFFICIAL_DATASETS[preset_name]
    if preset.eval_split is None:
        raise ValueError(f"{preset_name} does not define an eval split")
    dataset = load_split(preset, preset.eval_split, args.dataset_cache_dir)
    dataset = maybe_subset(dataset, max_samples=max_samples, seed=args.seed, shuffle=shuffle)
    source = (
        f"{preset_name}: {preset.path}/{preset.config or '-'} "
        f"({preset.eval_split}) <- {preset.source_url}"
    )
    return dataset, source


class WhisperBatchCollator:
    def __init__(self, processor: WhisperProcessor, max_label_tokens: int):
        # Сохраняет processor и параметры токенизации, нужные для сборки батчей.
        self.processor = processor
        self.max_label_tokens = max_label_tokens
        self.decoder_start_token_id = processor.tokenizer.bos_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Превращает список raw audio/text примеров в тензоры признаков, label-ы и служебную метаинформацию.
        audios = [item["audio"]["array"] for item in batch]
        texts = [normalise_text(item["text"]) for item in batch]

        features = self.processor.feature_extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )
        tokenised = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_label_tokens,
            return_tensors="pt",
        )

        labels = tokenised.input_ids.masked_fill(tokenised.attention_mask.ne(1), -100)
        if labels.size(1) > 0 and torch.all(labels[:, 0] == self.decoder_start_token_id):
            labels = labels[:, 1:]

        attention_mask = getattr(features, "attention_mask", None)
        return {
            "input_features": features.input_features,
            "attention_mask": attention_mask,
            "labels": labels,
            "texts": texts,
            "audio_seconds": [
                len(item["audio"]["array"]) / float(item["audio"]["sampling_rate"]) for item in batch
            ],
        }


def build_dataloader(
    dataset: Dataset,
    collator: WhisperBatchCollator,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    # Создает DataLoader с нужным collate_fn и настройками загрузки под CPU/GPU.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def prune_decoder_layers(model: WhisperForConditionalGeneration, keep_layers: list[int]) -> None:
    # Оставляет в student только выбранные decoder layers и обновляет конфиг модели.
    decoder_layers = model.model.decoder.layers
    num_layers = len(decoder_layers)
    if not keep_layers:
        raise ValueError("student layer list must not be empty")
    if any(index < 0 or index >= num_layers for index in keep_layers):
        raise ValueError(f"student layers must be within [0, {num_layers - 1}]")

    model.model.decoder.layers = nn.ModuleList(
        [copy.deepcopy(decoder_layers[index]) for index in keep_layers]
    )
    model.config.decoder_layers = len(keep_layers)
    model.model.decoder.config.decoder_layers = len(keep_layers)


def freeze_teacher(model: WhisperForConditionalGeneration) -> None:
    # Полностью замораживает teacher и переводит его в режим инференса.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False


def configure_student(
    model: WhisperForConditionalGeneration,
    *,
    freeze_encoder: bool,
    train_proj_out: bool,
    gradient_checkpointing: bool,
) -> None:
    # Настраивает student так, чтобы обучались только decoder и при необходимости proj_out.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if freeze_encoder:
        for parameter in model.model.encoder.parameters():
            parameter.requires_grad = False

    for parameter in model.model.decoder.parameters():
        parameter.requires_grad = True

    if train_proj_out:
        for parameter in model.proj_out.parameters():
            parameter.requires_grad = True

    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()


def build_teacher_student(
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[WhisperForConditionalGeneration, WhisperForConditionalGeneration]:
    # Загружает teacher, клонирует student, прореживает декодер и переносит обе модели на целевое устройство.
    teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher_model)
    student = copy.deepcopy(teacher)
    prune_decoder_layers(student, args.student_layers)
    configure_student(
        student,
        freeze_encoder=args.freeze_encoder,
        train_proj_out=args.train_proj_out,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    freeze_teacher(teacher)

    teacher_dtype = amp_dtype if device.type == "cuda" else torch.float32
    teacher.to(device=device, dtype=teacher_dtype)
    student.to(device=device)

    if args.freeze_encoder:
        student.model.encoder.to(dtype=teacher_dtype)

    return teacher, student


def masked_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    # Считает token-level KL только по валидным target-позициям, игнорируя padding и masked label-ы.
    mask = labels.ne(-100)
    if not mask.any():
        return student_logits.new_tensor(0.0)

    scaled_student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    scaled_teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    tokenwise_kl = F.kl_div(scaled_student, scaled_teacher, reduction="none").sum(dim=-1)
    return tokenwise_kl.masked_select(mask).mean() * (temperature**2)


class LiveDashboard:
    def __init__(self, output_dir: Path, *, enabled: bool, refresh_every: int):
        # Инициализирует JSONL-лог и matplotlib-панель для live-мониторинга обучения.
        self.enabled = enabled
        self.refresh_every = max(1, refresh_every)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.figure_path = self.output_dir / "live_metrics.png"
        self.history: dict[str, list[float]] = defaultdict(list)
        self.steps: list[int] = []
        self.figure = None
        self.axes = None

        if self.enabled:
            self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
            if os.environ.get("DISPLAY"):
                plt.ion()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        # Пишет очередную точку метрик в файл и обновляет внутреннюю историю для графиков.
        record = {"step": step, **metrics}
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

        self.steps.append(step)
        for key, value in metrics.items():
            self.history[key].append(float(value))

        if self.enabled and step % self.refresh_every == 0:
            self.render()

    def render(self) -> None:
        # Перерисовывает live-графики loss, optimizer state, validation quality и runtime.
        assert self.axes is not None
        for axis in self.axes.flat:
            axis.clear()

        self.axes[0, 0].plot(self.steps, self.history.get("loss", []), label="loss", color="#1f77b4")
        self.axes[0, 0].plot(
            self.steps,
            self.history.get("ce_loss", []),
            label="ce",
            color="#ff7f0e",
            alpha=0.85,
        )
        self.axes[0, 0].plot(
            self.steps,
            self.history.get("kl_loss", []),
            label="kl",
            color="#2ca02c",
            alpha=0.85,
        )
        self.axes[0, 0].set_title("Distillation Loss")
        self.axes[0, 0].legend()
        self.axes[0, 0].grid(alpha=0.25)

        self.axes[0, 1].plot(self.steps, self.history.get("lr", []), color="#d62728", label="lr")
        self.axes[0, 1].plot(
            self.steps,
            self.history.get("grad_norm", []),
            color="#9467bd",
            label="grad_norm",
        )
        self.axes[0, 1].set_title("Optimizer State")
        self.axes[0, 1].legend()
        self.axes[0, 1].grid(alpha=0.25)

        self.axes[1, 0].plot(
            self.steps,
            self.history.get("eval_wer", []),
            color="#8c564b",
            label="eval_wer",
        )
        self.axes[1, 0].plot(
            self.steps,
            self.history.get("eval_cer", []),
            color="#e377c2",
            label="eval_cer",
            alpha=0.85,
        )
        self.axes[1, 0].set_title("Validation Quality")
        self.axes[1, 0].legend()
        self.axes[1, 0].grid(alpha=0.25)

        self.axes[1, 1].plot(
            self.steps,
            self.history.get("samples_per_sec", []),
            color="#17becf",
            label="samples/s",
        )
        self.axes[1, 1].plot(
            self.steps,
            self.history.get("gpu_reserved_gb", []),
            color="#7f7f7f",
            label="gpu_reserved_gb",
            alpha=0.85,
        )
        self.axes[1, 1].set_title("Runtime")
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(alpha=0.25)

        self.figure.suptitle("Whisper Turbo Distillation Monitor", fontsize=15, fontweight="bold")
        self.figure.savefig(self.figure_path, dpi=160, bbox_inches="tight")
        if os.environ.get("DISPLAY"):
            self.figure.canvas.draw()
            self.figure.canvas.flush_events()

    def close(self) -> None:
        # Финализирует live-dashboard и сохраняет последнюю версию графиков.
        if self.enabled and self.figure is not None:
            self.render()
            plt.close(self.figure)


def save_checkpoint(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    output_dir: Path,
    name: str,
) -> Path:
    # Сохраняет веса student и processor в отдельную папку checkpoint-а.
    checkpoint_dir = output_dir / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    return checkpoint_dir


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, Any, torch.Tensor]:
    # Переносит входные признаки, маски и label-ы батча на нужное устройство.
    input_features = batch["input_features"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"]
    if attention_mask is not None:
        attention_mask = attention_mask.to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    return input_features, attention_mask, labels


@torch.no_grad()
def evaluate_model(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    dataloader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype,
    *,
    desc: str,
    warmup_batches: int = 0,
) -> dict[str, float]:
    # Запускает генерацию на датасете и считает WER, CER, latency и real-time factor.
    model.eval()
    predictions: list[str] = []
    references: list[str] = []
    total_wall = 0.0
    total_audio = 0.0
    total_samples = 0

    generation_kwargs = {
        "max_new_tokens": args.generation_max_new_tokens,
        "num_beams": args.generation_num_beams,
        "condition_on_prev_tokens": args.condition_on_prev_tokens,
        "language": args.language,
        "task": args.task,
    }

    for batch_index, batch in enumerate(tqdm(dataloader, desc=desc, leave=False), start=1):
        input_features, attention_mask, _ = move_batch_to_device(batch, device)

        if batch_index <= warmup_batches:
            with autocast_context(device, amp_dtype):
                _ = model.generate(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    **generation_kwargs,
                )
            continue

        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with autocast_context(device, amp_dtype):
            predicted_ids = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_wall += time.perf_counter() - started

        decoded = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        predictions.extend(normalise_text(text) for text in decoded)
        references.extend(batch["texts"])
        total_audio += sum(batch["audio_seconds"])
        total_samples += len(decoded)

    metrics = {
        "wer": float(wer(references, predictions)) if predictions else float("nan"),
        "cer": float(cer(references, predictions)) if predictions else float("nan"),
        "avg_latency_ms": 1000.0 * total_wall / max(total_samples, 1),
        "rtf": total_wall / max(total_audio, 1e-8),
        "samples": total_samples,
    }
    return metrics


def train(
    teacher: WhisperForConditionalGeneration,
    student: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    # Выполняет основной distillation loop с CE+KL loss, логированием, валидацией и сохранением student.
    trainable_parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    scaler = GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    dashboard = LiveDashboard(
        output_dir=output_dir,
        enabled=not args.disable_live_plots,
        refresh_every=args.plot_interval,
    )

    global_step = 0
    last_eval = {"eval_wer": float("nan"), "eval_cer": float("nan")}
    optimizer.zero_grad(set_to_none=True)
    student.train()

    progress = tqdm(total=total_steps, desc="optimizer steps")
    for epoch in range(1, args.epochs + 1):
        for micro_step, batch in enumerate(train_loader, start=1):
            started = time.perf_counter()
            input_features, attention_mask, labels = move_batch_to_device(batch, device)
            decoder_input_ids = shift_tokens_right(
                labels,
                student.config.pad_token_id,
                student.config.decoder_start_token_id,
            )
            decoder_attention_mask = decoder_input_ids.ne(student.config.pad_token_id).long()

            with autocast_context(device, amp_dtype):
                with torch.no_grad():
                    teacher_logits = teacher(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        decoder_input_ids=decoder_input_ids,
                        decoder_attention_mask=decoder_attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits

                student_logits = student(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits

                ce_loss = F.cross_entropy(
                    student_logits.reshape(-1, student_logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                kl_loss = masked_kl_divergence(
                    student_logits=student_logits,
                    teacher_logits=teacher_logits,
                    labels=labels,
                    temperature=args.kd_temperature,
                )
                loss = args.ce_weight * ce_loss + args.kl_weight * kl_loss

            loss_to_backprop = loss / args.gradient_accumulation_steps
            if scaler.is_enabled():
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            should_step = (
                micro_step % args.gradient_accumulation_steps == 0
                or micro_step == len(train_loader)
            )
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            progress.update(1)

            elapsed = time.perf_counter() - started
            samples_per_sec = input_features.size(0) / max(elapsed, 1e-8)
            gpu_reserved_gb = (
                torch.cuda.memory_reserved(device) / (1024**3) if device.type == "cuda" else 0.0
            )

            metrics = {
                "loss": float(loss.item()),
                "ce_loss": float(ce_loss.item()),
                "kl_loss": float(kl_loss.item()),
                "lr": float(scheduler.get_last_lr()[0]),
                "grad_norm": float(grad_norm),
                "samples_per_sec": float(samples_per_sec),
                "gpu_reserved_gb": float(gpu_reserved_gb),
            }

            if args.eval_interval > 0 and global_step % args.eval_interval == 0:
                last_eval = evaluate_model(
                    student,
                    processor,
                    eval_loader,
                    args,
                    device,
                    amp_dtype,
                    desc=f"eval @ step {global_step}",
                )
                metrics["eval_wer"] = last_eval["wer"]
                metrics["eval_cer"] = last_eval["cer"]
                student.train()
            else:
                metrics.update(last_eval)

            dashboard.log(global_step, metrics)

            if global_step % args.log_interval == 0:
                tqdm.write(
                    f"[step {global_step:05d}/{total_steps:05d}] "
                    f"loss={metrics['loss']:.4f} "
                    f"ce={metrics['ce_loss']:.4f} "
                    f"kl={metrics['kl_loss']:.4f} "
                    f"lr={metrics['lr']:.2e} "
                    f"grad={metrics['grad_norm']:.3f} "
                    f"eval_wer={metrics['eval_wer']:.4f}"
                )

            if args.save_interval > 0 and global_step % args.save_interval == 0:
                save_checkpoint(student, processor, output_dir, f"checkpoint-step-{global_step}")

        tqdm.write(f"[epoch {epoch}/{args.epochs}] completed")

    progress.close()
    dashboard.close()
    save_checkpoint(student, processor, output_dir, "final_student")
    return last_eval


def print_run_header(
    args: argparse.Namespace,
    train_source_messages: list[str],
    eval_source: str,
    latency_source: str,
    student: WhisperForConditionalGeneration,
) -> None:
    # Печатает компактную сводку по конфигурации запуска, модели и источникам данных.
    print("\n== Whisper Turbo Distillation ==")
    print(f"teacher                : {args.teacher_model}")
    print(f"student kept layers    : {args.student_layers}")
    print("recipe                 : freeze encoder, train decoder + proj_out, loss=0.5 CE + 0.5 KL")
    print(f"device / dtype         : {args.device} / {args.dtype}")
    print(f"batch / accum          : {args.train_batch_size} / {args.gradient_accumulation_steps}")
    print(f"epochs / lr            : {args.epochs} / {args.learning_rate}")
    print(f"trainable params       : {count_parameters(student, trainable_only=True):,}")
    print(f"student total params   : {count_parameters(student):,}")
    print("train sources          :")
    for message in train_source_messages:
        print(f"  - {message}")
    print(f"eval source            : {eval_source}")
    print(f"latency source         : {latency_source}")
    print("")


def main() -> None:
    # Собирает все этапы вместе: аргументы, данные, модели, обучение и финальный benchmark.
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(serialize_args(args), handle, indent=2, ensure_ascii=True)

    set_random_seed(args.seed)
    device = torch.device(args.device)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested, but CUDA is not available.")
    amp_dtype = dtype_from_name(args.dtype)

    processor = WhisperProcessor.from_pretrained(args.teacher_model)
    processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)

    train_dataset, train_source_messages = load_training_dataset(args)
    eval_dataset, eval_source = load_eval_dataset(
        args.eval_preset,
        args,
        max_samples=args.max_eval_samples,
        shuffle=False,
    )
    try:
        latency_dataset, latency_source = load_eval_dataset(
            args.latency_preset,
            args,
            max_samples=args.max_latency_samples,
            shuffle=False,
        )
    except Exception as exc:
        latency_dataset = eval_dataset
        latency_source = f"{eval_source} (fallback after latency preset error: {exc})"

    collator = WhisperBatchCollator(processor, max_label_tokens=args.max_label_tokens)
    pin_memory = device.type == "cuda"
    train_loader = build_dataloader(
        train_dataset,
        collator,
        args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = build_dataloader(
        eval_dataset,
        collator,
        args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    latency_loader = build_dataloader(
        latency_dataset,
        collator,
        args.latency_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    teacher, student = build_teacher_student(args, device, amp_dtype)
    print_run_header(args, train_source_messages, eval_source, latency_source, student)

    last_eval = train(
        teacher,
        student,
        processor,
        train_loader,
        eval_loader,
        args,
        output_dir,
        device,
        amp_dtype,
    )

    teacher_eval = evaluate_model(
        teacher,
        processor,
        eval_loader,
        args,
        device,
        amp_dtype,
        desc="teacher eval",
    )
    student_eval = evaluate_model(
        student,
        processor,
        eval_loader,
        args,
        device,
        amp_dtype,
        desc="student eval",
    )
    teacher_latency = evaluate_model(
        teacher,
        processor,
        latency_loader,
        args,
        device,
        amp_dtype,
        desc="teacher latency",
        warmup_batches=2,
    )
    student_latency = evaluate_model(
        student,
        processor,
        latency_loader,
        args,
        device,
        amp_dtype,
        desc="student latency",
        warmup_batches=2,
    )

    summary = {
        "last_eval": last_eval,
        "teacher_baseline": {
            "eval": teacher_eval,
            "latency": teacher_latency,
        },
        "student_distilled": {
            "eval": student_eval,
            "latency": student_latency,
        },
        "teacher_model": args.teacher_model,
        "student_layers": args.student_layers,
    }
    with (output_dir / "benchmark_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print("== Final Benchmark ==")
    print(
        f"baseline turbo | WER={teacher_eval['wer']:.4f} "
        f"CER={teacher_eval['cer']:.4f} "
        f"latency={teacher_latency['avg_latency_ms']:.2f} ms/sample "
        f"RTF={teacher_latency['rtf']:.4f}"
    )
    print(
        f"student 2-layer| WER={student_eval['wer']:.4f} "
        f"CER={student_eval['cer']:.4f} "
        f"latency={student_latency['avg_latency_ms']:.2f} ms/sample "
        f"RTF={student_latency['rtf']:.4f}"
    )
    print(f"artifacts saved to     : {output_dir.resolve()}")
    print(f"live plot              : {(output_dir / 'live_metrics.png').resolve()}")
    print(f"json summary           : {(output_dir / 'benchmark_summary.json').resolve()}")


if __name__ == "__main__":
    main()
