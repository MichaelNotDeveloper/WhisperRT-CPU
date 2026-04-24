import time
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from audio_datasets import AudioTextDataset
from utils import (
    BenchmarkResult,
    ModuleTimer,
    asr_metrics,
    plot_benchmarks,
    plot_profiler_averages,
)


# Models are expected to expose `model.encoder` and `model.decoder`.
@dataclass
class ModelsForBenchmark:
    BaseWhisper: nn.Module | None = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base"
    )
    LargeWhisper: nn.Module | None = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3"
    ).to(dtype=torch.float)


@dataclass
class ProcessingOptions:
    BaseWhisper: Any = WhisperProcessor.from_pretrained("openai/whisper-base")
    LargeWhisper: Any = WhisperProcessor.from_pretrained("openai/whisper-large-v3")


class Benchmark:
    def __init__(
        self,
        dataset_name: str,
        models: dict[str, nn.Module],
        device: Literal["CPU", "GPU"],
    ):
        self.models = models
        self.processors = {}
        self.device = "cpu" if device == "CPU" else "cuda"
        self.dataset = AudioTextDataset(dataset_name)
        self.profiler_activities = (
            [ProfilerActivity.CPU]
            if device == "CPU"
            else [ProfilerActivity.CUDA, ProfilerActivity.CPU]
        )

        for field in fields(ModelsForBenchmark):
            self.models[field.name] = getattr(ModelsForBenchmark, field.name)

        for field in fields(ProcessingOptions):
            self.processors[field.name] = getattr(ProcessingOptions, field.name)

        for model in self.models.values():
            model = model.to(self.device)

    def run(self, model_name: str, sample_size: int = 10) -> BenchmarkResult:
        model = self.models[model_name]
        processor = self.processors[model_name]

        wer_hist = []
        cer_hist = []
        generated_texts = []
        original_texts = []

        encoder_timer = ModuleTimer()
        decoder_timer = ModuleTimer()
        encoder_timer.wrap(model.model.encoder)
        decoder_timer.wrap(model.model.decoder)
        processor_speed = []
        with profile(activities=self.profiler_activities, record_shapes=True) as prof:
            with record_function("model_inference"):
                for audio, text in tqdm(
                    self.dataset.take(sample_size),
                    total=sample_size,
                    desc=f"Benchmarking {model_name}",
                ):
                    processor_start = time.perf_counter()
                    inputs = processor(
                        audio["array"],
                        sampling_rate=audio["sampling_rate"],
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    processor_speed.append(time.perf_counter() - processor_start)

                    input_features = inputs.input_features.to(self.device)
                    attention_mask = inputs.attention_mask.to(self.device)

                    tokens = model.generate(
                        input_features=input_features,
                        attention_mask=attention_mask,
                        language="ru",  # или "ru" для golos
                        task="transcribe",
                    )
                    result = processor.batch_decode(tokens, skip_special_tokens=True)[0]

                    metrics = asr_metrics(result, text)
                    wer_hist.append(metrics["wer"])
                    cer_hist.append(metrics["cer"])
                    generated_texts.append(result)
                    original_texts.append(text)

        return BenchmarkResult(
            wer_history=wer_hist,
            cer_history=cer_hist,
            generated_texts=generated_texts,
            original_texts=original_texts,
            encoder_speed=encoder_timer.times,
            decoder_speed=decoder_timer.times,
            processor_speed=processor_speed,
            profiler=prof.key_averages().table(
                sort_by=("self_cpu_time_total" if self.device == "cpu" else "self_cuda_time_total"),
                row_limit=10,
            ),
        )

    def get_models(self):
        return self.models


if __name__ == "__main__":
    results_base = Benchmark(
        dataset_name="golos",
        models={},
        device="CPU",
    ).run("BaseWhisper", sample_size=20)
    results_large = Benchmark(
        dataset_name="golos",
        models={},
        device="CPU",
    ).run("LargeWhisper", sample_size=20)
    results = {"BaseWhisper": results_base, "LargeWhisper": results_large}
    plot_benchmarks(results, "./plots.png")
    plot_profiler_averages(results, "./profiler_plot.png")
