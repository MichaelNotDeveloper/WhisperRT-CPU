import gc
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torch


def clean_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_asr_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е")

    normalized_chars = []
    for char in text:
        category = unicodedata.category(char)
        if category == "Mn":
            continue
        if category.startswith(("P", "S")) or char == "_":
            normalized_chars.append(" ")
            continue
        normalized_chars.append(char)

    return _WHITESPACE_RE.sub(" ", "".join(normalized_chars)).strip()


def asr_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    ref_tr = _normalize_asr_text(reference)
    hyp_tr = _normalize_asr_text(hypothesis)
    out = jiwer.process_words(ref_tr, hyp_tr)
    wer = out.wer
    cer = jiwer.cer(ref_tr, hyp_tr)
    return {
        "wer": wer,
        "cer": cer,
    }


class ModuleTimer:
    def __init__(self, device_type: str = "cpu"):
        self.times = []
        self.device_type = device_type

    def wrap(self, module):
        original_forward = module.forward

        @wraps(original_forward)
        def timed_forward(*args, **kwargs):
            if self.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = original_forward(*args, **kwargs)

            if self.device_type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()

            t1 = time.perf_counter()
            self.times.append(t1 - t0)
            return out

        module.forward = timed_forward
        return original_forward


@dataclass
class BenchmarkResult:
    wer_history: list[float]
    cer_history: list[float]
    generated_texts: list[str]
    original_texts: list[str]
    audio_time_ratio: list[float]
    encoder_speed: list[float]
    decoder_speed: list[float]
    full_generation_step_speed: list[float] = field(default_factory=list)
    processor_speed: list[float] = field(default_factory=list)
    successful_predictions: list[int] = field(default_factory=list)
    proposed_predictions: list[int] = field(default_factory=list)
    prediction_success_ratio: list[float] = field(default_factory=list)
    speculative_steps: list[int] = field(default_factory=list)
    profiler: Any = None


def _mean_ci(values: list[float], z_value: float = 1.96) -> tuple[float, float, int]:
    values_array = np.asarray(values, dtype=float)
    values_array = values_array[np.isfinite(values_array)]
    if values_array.size == 0:
        return float("nan"), 0.0, 0
    mean = float(values_array.mean())
    if values_array.size == 1:
        return mean, 0.0, 1
    stderr = float(values_array.std(ddof=1)) / math.sqrt(values_array.size)
    return mean, z_value * stderr, int(values_array.size)


def _text_stats(texts: list[str]) -> tuple[float, float]:
    if not texts:
        return 0.0, 0.0
    lengths = np.asarray([len(text) for text in texts], dtype=float)
    return float(lengths.mean()), float(lengths.std(ddof=0))


def _plot_metric(ax, results, attr, title, ylabel, color, *, show_ci=False, lower_is_better=True):
    names = list(results)
    means, cis, counts = [], [], []
    for name in names:
        mean, ci, count = _mean_ci(getattr(results[name], attr))
        means.append(mean)
        cis.append(ci if show_ci else 0.0)
        counts.append(count)

    x = np.arange(len(names))
    bars = ax.bar(
        x,
        means,
        yerr=cis if any(ci > 0 for ci in cis) else None,
        capsize=6,
        color=color,
        edgecolor="#2b2b2b",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_title(title, fontsize=13, fontweight="semibold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, names)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.98,
        "lower is better" if lower_is_better else "higher is better",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#5a5a5a",
    )

    finite_tops = [mean + ci for mean, ci in zip(means, cis, strict=False) if np.isfinite(mean)]
    max_top = max(finite_tops, default=1.0)
    ax.set_ylim(0, max(max_top * 1.3, 0.1))

    labels = []
    for mean, ci, count in zip(means, cis, counts, strict=False):
        if not np.isfinite(mean):
            labels.append("")
            continue
        label = f"{mean:.4f}\nn={count}"
        if show_ci:
            label += f"\n±{ci:.4f}"
        labels.append(label)
    ax.bar_label(bars, labels=labels, padding=4, fontsize=9)


def plot_benchmarks(
    results: dict[str, BenchmarkResult],
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Строит сравнительные графики по всем числовым benchmark-метрикам.

    На каждом графике:
    - одна статистика
    - несколько моделей
    - подписаны реальные средние значения
    - для декодера показан 95% доверительный интервал
    """

    if not results:
        raise ValueError("results is empty")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)
        fig.suptitle("Benchmark Comparison", fontsize=16, fontweight="bold")

        _plot_metric(
            axes[0, 0],
            results,
            "wer_history",
            "WER",
            "Word Error Rate",
            "#4C78A8",
        )
        _plot_metric(
            axes[0, 1],
            results,
            "cer_history",
            "CER",
            "Character Error Rate",
            "#F58518",
        )
        _plot_metric(
            axes[0, 2],
            results,
            "audio_time_ratio",
            "Runtime / Audio Duration (RTF)",
            "Ratio",
            "#B279A2",
        )
        _plot_metric(
            axes[1, 0],
            results,
            "encoder_speed",
            "Encoder Speed",
            "Seconds per forward",
            "#54A24B",
        )
        _plot_metric(
            axes[1, 1],
            results,
            "decoder_speed",
            "Decoder Speed (95% CI)",
            "Seconds per generation step",
            "#E45756",
            show_ci=True,
        )
        _plot_metric(
            axes[1, 2],
            results,
            "full_generation_step_speed",
            "Full Generation Step Speed",
            "Seconds per full generation step",
            "#72B7B2",
        )

        summary_lines = []
        for name, result in results.items():
            gen_mean, _ = _text_stats(result.generated_texts)
            ref_mean, _ = _text_stats(result.original_texts)
            summary = (
                f"{name}: samples={len(result.wer_history)}, "
                f"avg generated chars={gen_mean:.1f}, avg reference chars={ref_mean:.1f}"
            )
            proposed_total = sum(result.proposed_predictions)
            if proposed_total > 0:
                successful_total = sum(result.successful_predictions)
                summary += (
                    f", mtp_success={successful_total}/{proposed_total} "
                    f"({successful_total / proposed_total:.1%})"
                )
            summary_lines.append(summary)
        fig.text(
            0.5,
            0.965,
            " | ".join(summary_lines),
            ha="center",
            va="top",
            fontsize=9,
            color="#555555",
        )

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes


def _parse_time_to_ms(value: str) -> float:
    value = value.strip()
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*(us|ms|s|ns)", value)
    if match is None:
        raise ValueError(f"Unsupported profiler time format: {value!r}")

    number = float(match.group(1))
    unit = match.group(2)
    if unit == "s":
        return number * 1000.0
    if unit == "ms":
        return number
    if unit == "us":
        return number / 1000.0
    return number / 1_000_000.0


def _parse_profiler_table(table: str) -> tuple[str, list[tuple[str, float]]]:
    lines = [line.strip() for line in table.splitlines() if line.strip()]
    header = next((line for line in lines if "Name" in line and "avg" in line), None)
    if header is None:
        raise ValueError("Profiler table header was not found")

    columns = re.split(r"\s{2,}", header)
    avg_column = next((col for col in columns if "time avg" in col.lower()), None)
    if avg_column is None:
        raise ValueError("Profiler average-time column was not found")

    avg_index = columns.index(avg_column)
    rows: list[tuple[str, float]] = []
    for line in lines[lines.index(header) + 1 :]:
        if set(line) == {"-"}:
            continue
        parts = re.split(r"\s{2,}", line)
        if len(parts) <= avg_index or parts[0] == "Self CPU time total:":
            continue
        name = parts[0]
        try:
            rows.append((name, _parse_time_to_ms(parts[avg_index])))
        except ValueError:
            continue

    return avg_column, rows


def plot_profiler_averages(
    results: dict[str, BenchmarkResult] | None,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure | None, plt.Axes | None]:
    """Строит отдельный grouped barplot по среднему времени операций из torch.profiler."""

    if not results:
        return None, None

    parsed_results: dict[str, dict[str, float]] = {}
    avg_column_name = None
    op_names: list[str] = []
    for model_name, result in results.items():
        if not result.profiler:
            continue
        avg_column_name, rows = _parse_profiler_table(result.profiler)
        parsed_results[model_name] = {name: value for name, value in rows}
        for name, _ in rows:
            if name not in op_names:
                op_names.append(name)

    if not parsed_results or not op_names:
        return None, None

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig_height = max(6, 0.5 * len(op_names) + 2)
        fig, ax = plt.subplots(figsize=(16, fig_height), constrained_layout=True)
        fig.suptitle("Profiler Average Time by Operation", fontsize=16, fontweight="bold")

        y = np.arange(len(op_names))
        model_names = list(parsed_results)
        bar_height = 0.8 / max(len(model_names), 1)
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

        for index, (model_name, color) in enumerate(zip(model_names, colors, strict=False)):
            values = [parsed_results[model_name].get(op_name, 0.0) for op_name in op_names]
            offset = (index - (len(model_names) - 1) / 2) * bar_height
            bars = ax.barh(
                y + offset,
                values,
                height=bar_height,
                label=model_name,
                color=color,
                edgecolor="#2b2b2b",
                linewidth=0.8,
                alpha=0.9,
            )
            for bar, value in zip(bars, values, strict=False):
                if value <= 0:
                    continue
                ax.text(
                    bar.get_width() + max(max(values, default=1.0) * 0.01, 0.02),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.3f} ms",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

        ax.set_yticks(y, op_names)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(f"{avg_column_name} (ms)")
        ax.set_ylabel("Profiler operation name")
        ax.grid(axis="x", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="lower right")

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, ax
