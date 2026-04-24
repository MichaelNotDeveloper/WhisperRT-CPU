import gc
import math
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torch
from jiwer import Compose, RemoveMultipleSpaces, Strip, ToLowerCase


def clean_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def seed_everything(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def asr_metrics(hypothesis: str, reference: str) -> dict[str, float]:
    tr = Compose([ToLowerCase(), RemoveMultipleSpaces(), Strip()])
    ref_tr = tr(reference)
    hyp_tr = tr(hypothesis)
    out = jiwer.process_words(ref_tr, hyp_tr)
    wer = out.wer
    cer = jiwer.cer(ref_tr, hyp_tr)
    return {
        "wer": wer,
        "cer": cer,
    }

class ModuleTimer:
    def __init__(self):
        self.times = []

    def wrap(self, module):
        original_forward = module.forward

        @wraps(original_forward)
        def timed_forward(*args, **kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            out = original_forward(*args, **kwargs)

            if torch.cuda.is_available():
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
    encoder_speed: list[float]
    decoder_speed: list[float]


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

    max_value = max((abs(value) for value in means if np.isfinite(value)), default=1.0)
    label_y_offset = max(max_value * 0.03, 0.01)
    for bar, mean, ci, count in zip(bars, means, cis, counts, strict=False):
        if not np.isfinite(mean):
            continue
        label = f"{mean:.4f}\nn={count}"
        if show_ci:
            label += f"\n±{ci:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


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
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
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

        summary_lines = []
        for name, result in results.items():
            gen_mean, _ = _text_stats(result.generated_texts)
            ref_mean, _ = _text_stats(result.original_texts)
            summary_lines.append(
                f"{name}: samples={len(result.wer_history)}, "
                f"avg generated chars={gen_mean:.1f}, avg reference chars={ref_mean:.1f}"
            )
        fig.text(
            0.5,
            0.005,
            "\n".join(summary_lines),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#444444",
        )

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig, axes
