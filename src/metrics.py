import time
from dataclasses import dataclass, fields
from typing import Any, Literal

from contextlib import nullcontext
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import WhisperProcessor

from audio_datasets import AudioTextDataset
from baseline_models import BaselineTurbo, TorchCompileTurboParams
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
    TurboWhisper: nn.Module | None = BaselineTurbo()()
    #LargeWhisper: nn.Module | None = BaselineLarge()()
    CompileWhipser: nn.Module | None = TorchCompileTurboParams()()

@dataclass
class ProcessingOptions:
    TurboWhisper: Any = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
    #LargeWhisper: Any = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    CompileWhipser: Any = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

class Benchmark:
    def __init__(
        self,
        dataset_name: str,
        models: dict[str, nn.Module],
        device: Literal["CPU", "GPU"],
        profiler: bool = True
    ):
        self.models = models
        self.processors = {}
        self._warmed_up = set()
        self.device = "cpu" if device == "CPU" else "cuda"
        self.dataset = AudioTextDataset(dataset_name)
        self.profiler_activities = (
            [ProfilerActivity.CPU]
            if device == "CPU"
            else [ProfilerActivity.CUDA, ProfilerActivity.CPU]
        )
        self.profiler_state = profiler
        self.profiler = profile(activities=self.profiler_activities, record_shapes=True) if profiler else nullcontext()
        self.record_func = record_function("model_inference") if profiler else nullcontext() 

        for field in fields(ModelsForBenchmark):
            self.models[field.name] = getattr(ModelsForBenchmark, field.name)

        for field in fields(ProcessingOptions):
            self.processors[field.name] = getattr(ProcessingOptions, field.name)

        for model in self.models.values():
            model = model.to(self.device)

    def warmup(self, model_name: str):
        if model_name in self._warmed_up:
            return

        sample = self.dataset.take(1)
        if not sample:
            return

        model = self.models[model_name]
        processor = self.processors[model_name]
        encoder_timer = ModuleTimer()
        decoder_timer = ModuleTimer()
        original_encoder = encoder_timer.wrap(model.model.encoder)
        original_decoder = decoder_timer.wrap(model.model.decoder)

        try:
            audio, _ = sample[0]
            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
                return_attention_mask=True,
            )
            with torch.inference_mode():
                model.generate(
                    input_features=inputs.input_features.to(self.device),
                    attention_mask=inputs.attention_mask.to(self.device),
                    language="ru",
                    task="transcribe",
                )
        finally:
            model.model.encoder.forward = original_encoder
            model.model.decoder.forward = original_decoder

        self._warmed_up.add(model_name)

    def run(self, model_name: str, sample_size: int = 10) -> BenchmarkResult:
        model = self.models[model_name]
        processor = self.processors[model_name]

        wer_hist = []
        cer_hist = []
        generated_texts = []
        original_texts = []
        audio_time_ratio = []
        
        self.warmup(model_name)

        encoder_timer = ModuleTimer()
        decoder_timer = ModuleTimer()
        encoder_timer.wrap(model.model.encoder)
        decoder_timer.wrap(model.model.decoder)
        processor_speed = []
        with self.profiler as prof:
            with self.record_func:
                for audio, text in tqdm(
                    self.dataset.take(sample_size),
                    total=sample_size,
                    desc=f"Benchmarking {model_name}",
                ):
                    sample_start = time.perf_counter()
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
                        language="ru",
                        task="transcribe",
                    )
                    result = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                    sample_runtime = time.perf_counter() - sample_start
                    audio_duration = len(audio["array"]) / audio["sampling_rate"]
                    audio_time_ratio.append(sample_runtime / audio_duration)
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
            audio_time_ratio=audio_time_ratio,
            encoder_speed=encoder_timer.times,
            decoder_speed=decoder_timer.times,
            processor_speed=processor_speed,
            profiler=prof.key_averages().table(
                sort_by=("self_cpu_time_total" if self.device == "cpu" else "self_cuda_time_total"),
                row_limit=10,
            ) if self.profiler_state else None,
        )

    def get_models(self):
        return self.models


if __name__ == "__main__":
    bench = Benchmark(
        dataset_name="earnings22",
        models={},
        device="CPU",
        profiler=False
    )
    results_large = bench.run("CompileWhipser", sample_size=20)
    results_base = bench.run("TurboWhisper", sample_size=20)
    results = {"TurboWhisper": results_base, "CompileWhipser": results_large}
    plot_benchmarks(results, "./plots.png")
    plot_profiler_averages(results, "./profiler_plot.png")
