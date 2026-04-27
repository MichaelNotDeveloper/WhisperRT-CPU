import time
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm
from transformers import WhisperFeatureExtractor, WhisperProcessor

from audio_datasets import AudioTextDataset
from baseline_models import (
    BaselineLarge,  # noqa: F401
    BaselineSmall,  # noqa: F401
    BaselineTurboParams,
    # DEFAULT_MTP_CHECKPOINT,
    # MultiTokenTurboPredictor,
    PrunedTurboDecoder,
    TorchCompileTurboParams,  # noqa: F401
)
from utils import (
    BenchmarkResult,
    ModuleTimer,
    asr_metrics,
    plot_benchmarks,
    plot_profiler_averages,
)

PRUNED_TURBO_CHECKPOINT = "2DecoderModelWeights"
DEFAULT_TASK = "transcribe"
DEFAULT_MAX_NEW_TOKENS = None
DATASET_LANGUAGES = {
    "earnings22": "en",
    "librispeech": "en",
    "golos": "ru",
}


def load_whisper_processor(
    source: str,
    expected_feature_size: int | None = None,
) -> WhisperProcessor:
    processor = WhisperProcessor.from_pretrained(source)

    if expected_feature_size is None:
        return processor

    if processor.feature_extractor.feature_size == expected_feature_size:
        return processor

    feature_kwargs = processor.feature_extractor.to_dict()
    feature_kwargs["feature_size"] = expected_feature_size
    feature_extractor = WhisperFeatureExtractor(**feature_kwargs)
    return WhisperProcessor(feature_extractor=feature_extractor, tokenizer=processor.tokenizer)


# Models are expected to expose `model.encoder` and `model.decoder`.
@dataclass
class ModelsForBenchmark:
    # BaseWhisper: nn.Module | None = BaselineSmall()()
    TurboWhisper: nn.Module | None = BaselineTurboParams()()
    # LargeV3Whisper: nn.Module | None = BaselineLarge()()
    # CompileWhipser: nn.Module | None = TorchCompileTurboParams()()
    PrunedTurbo2Decoder: nn.Module | None = PrunedTurboDecoder(PRUNED_TURBO_CHECKPOINT)()
    # MTPWhisperTurbo: nn.Module | None = MultiTokenTurboPredictor(DEFAULT_MTP_CHECKPOINT)


@dataclass
class ProcessingOptions:
    # BaseWhisper: str = "openai/whisper-base"
    TurboWhisper: str = "openai/whisper-large-v3-turbo"
    # LargeV3Whisper: str = "openai/whisper-large-v3"
    # CompileWhipser: str = "openai/whisper-large-v3-turbo"
    PrunedTurbo2Decoder: str = PRUNED_TURBO_CHECKPOINT
    # MTPWhisperTurbo: str = "openai/whisper-large-v3-turbo"


class Benchmark:
    def __init__(
        self,
        dataset_name: str,
        models: dict[str, nn.Module],
        device: Literal["CPU", "GPU"],
        profiler: bool = True,
    ):
        self.models = models
        self.processors = {}
        self._warmed_up = set()
        self.device = "cpu" if device == "CPU" else "cuda"
        self.dataset = AudioTextDataset(dataset_name)
        self.language = DATASET_LANGUAGES.get(dataset_name)
        self.profiler_activities = (
            [ProfilerActivity.CPU]
            if device == "CPU"
            else [ProfilerActivity.CUDA, ProfilerActivity.CPU]
        )
        self.profiler_state = profiler
        self.profiler = (
            profile(activities=self.profiler_activities, record_shapes=True)
            if profiler
            else nullcontext()
        )
        self.record_func = record_function("model_inference") if profiler else nullcontext()
        self.measure_module_timing = True

        for field in fields(ModelsForBenchmark):
            model = getattr(ModelsForBenchmark, field.name)
            if field.name not in self.models:
                self.models[field.name] = model

        for field in fields(ProcessingOptions):
            if field.name not in self.models:
                continue
            processor_source = getattr(ProcessingOptions, field.name)
            expected_feature_size = None
            if field.name == "PrunedTurbo2Decoder":
                expected_feature_size = self.models[field.name].config.num_mel_bins

            self.processors[field.name] = load_whisper_processor(
                processor_source,
                expected_feature_size=expected_feature_size,
            )

        for model in self.models.values():
            model.to(self.device)

    def _generate_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "task": DEFAULT_TASK,
            "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        }
        if self.language is not None:
            kwargs["language"] = self.language
        return kwargs

    def warmup(self, model_name: str):
        if model_name in self._warmed_up:
            return

        sample = self.dataset.take(1)
        if not sample:
            return

        model = self.models[model_name]
        processor = self.processors[model_name]
        original_encoder = None
        original_decoder = None
        if self.measure_module_timing:
            encoder_timer = ModuleTimer(device_type=self.device)
            decoder_timer = ModuleTimer(device_type=self.device)
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
                self._generate(
                    model,
                    processor,
                    inputs.input_features.to(self.device),
                    inputs.attention_mask.to(self.device),
                )
        finally:
            if original_encoder is not None:
                model.model.encoder.forward = original_encoder
            if original_decoder is not None:
                model.model.decoder.forward = original_decoder

        self._warmed_up.add(model_name)

    def _generate(self, model, processor, input_features, attention_mask):
        kwargs = self._generate_kwargs()
        if getattr(model, "requires_processor", False):
            kwargs["processor"] = processor
        return model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            **kwargs,
        )

    def _generation_stats(self, model) -> dict[str, Any]:
        stats = getattr(model, "last_generation_stats", None)
        if not isinstance(stats, dict):
            return {}
        out = {
            "successful_predictions": int(stats.get("successful_predictions", 0)),
            "proposed_predictions": int(stats.get("proposed_predictions", 0)),
            "accepted_tokens": int(stats.get("accepted_tokens", 0)),
            "steps": int(stats.get("steps", 0)),
        }
        if "full_generation_step_times" in stats:
            out["full_generation_step_times"] = list(stats.get("full_generation_step_times", []))
        return out

    def run(
        self,
        model_name: str,
        sample_size: int = 10,
        print_predictions: bool = False,
    ) -> BenchmarkResult:
        model = self.models[model_name]
        processor = self.processors[model_name]

        wer_hist = []
        cer_hist = []
        generated_texts = []
        original_texts = []
        audio_time_ratio = []
        successful_predictions = []
        proposed_predictions = []
        prediction_success_ratio = []
        speculative_steps = []
        full_generation_step_speed = []

        self.warmup(model_name)

        encoder_timer = ModuleTimer(device_type=self.device)
        decoder_timer = ModuleTimer(device_type=self.device)
        original_encoder = None
        original_decoder = None
        if self.measure_module_timing:
            original_encoder = encoder_timer.wrap(model.model.encoder)
            original_decoder = decoder_timer.wrap(model.model.decoder)
        processor_speed = []
        try:
            with self.profiler as prof:
                with self.record_func:
                    with torch.inference_mode():
                        for sample_index, (audio, text) in enumerate(
                            tqdm(
                                self.dataset.take(sample_size),
                                total=sample_size,
                                desc=f"Benchmarking {model_name}",
                            ),
                            start=1,
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

                            decoder_calls_before = len(decoder_timer.times)
                            tokens = self._generate(
                                model,
                                processor,
                                input_features,
                                attention_mask,
                            )
                            decoder_calls_after = len(decoder_timer.times)
                            generation_stats = self._generation_stats(model)
                            result = processor.batch_decode(tokens, skip_special_tokens=True)[0]
                            sample_runtime = time.perf_counter() - sample_start
                            audio_duration = len(audio["array"]) / audio["sampling_rate"]
                            audio_time_ratio.append(sample_runtime / audio_duration)

                            if "full_generation_step_times" in generation_stats:
                                full_generation_step_speed.extend(
                                    generation_stats["full_generation_step_times"]
                                )
                            else:
                                full_generation_step_speed.extend(
                                    decoder_timer.times[decoder_calls_before:decoder_calls_after]
                                )

                            if generation_stats:
                                successful = generation_stats["successful_predictions"]
                                proposed = generation_stats["proposed_predictions"]
                                successful_predictions.append(successful)
                                proposed_predictions.append(proposed)
                                speculative_steps.append(generation_stats["steps"])
                                prediction_success_ratio.append(
                                    successful / proposed if proposed else 0.0
                                )

                            if print_predictions:
                                tqdm.write(f"[{model_name} #{sample_index}] target: {text}")
                                tqdm.write(f"[{model_name} #{sample_index}] prediction: {result}")
                                if generation_stats:
                                    tqdm.write(
                                        f"[{model_name} #{sample_index}] mtp success: "
                                        f"{generation_stats['successful_predictions']}/"
                                        f"{generation_stats['proposed_predictions']} "
                                        f"(steps={generation_stats['steps']})"
                                    )

                            metrics = asr_metrics(result, text)
                            wer_hist.append(metrics["wer"])
                            cer_hist.append(metrics["cer"])
                            generated_texts.append(result)
                            original_texts.append(text)
        finally:
            if original_encoder is not None:
                model.model.encoder.forward = original_encoder
            if original_decoder is not None:
                model.model.decoder.forward = original_decoder

        if proposed_predictions:
            successful_total = sum(successful_predictions)
            proposed_total = sum(proposed_predictions)
            tqdm.write(
                f"[{model_name}] mtp success total: {successful_total}/{proposed_total} "
                f"({successful_total / proposed_total:.1%})"
            )

        return BenchmarkResult(
            wer_history=wer_hist,
            cer_history=cer_hist,
            generated_texts=generated_texts,
            original_texts=original_texts,
            audio_time_ratio=audio_time_ratio,
            encoder_speed=encoder_timer.times,
            decoder_speed=decoder_timer.times,
            full_generation_step_speed=full_generation_step_speed,
            processor_speed=processor_speed,
            successful_predictions=successful_predictions,
            proposed_predictions=proposed_predictions,
            prediction_success_ratio=prediction_success_ratio,
            speculative_steps=speculative_steps,
            profiler=(
                prof.key_averages().table(
                    sort_by=(
                        "self_cpu_time_total" if self.device == "cpu" else "self_cuda_time_total"
                    ),
                    row_limit=10,
                )
                if self.profiler_state
                else None
            ),
        )

    def get_models(self):
        return self.models


if __name__ == "__main__":
    bench1 = Benchmark(
        dataset_name="earnings22",
        models={},
        device="CPU",
        profiler=False,
    )
    bench2 = Benchmark(
        dataset_name="librispeech",
        models={},
        device="CPU",
        profiler=False,
    )
    # results_small = bench1.run("BaseWhisper", sample_size=20)
    results_base = bench1.run("TurboWhisper", sample_size=20)
    # results_large_v3 = bench1.run("LargeV3Whisper", sample_size=20)
    # results_large = bench1.run("CompileWhipser", sample_size=20)
    results_pruned = bench2.run("PrunedTurbo2Decoder", sample_size=20, print_predictions=True)
    results = {
        # "BaseWhisper": results_small,
        "TurboWhisper": results_base,
        # "LargeV3Whisper": results_large_v3,
        # "CompileWhipser": results_large,
        "PrunedTurbo2Decoder": results_pruned,
        # "MTPWhisperTurbo": bench2.run("MTPWhisperTurbo", sample_size=50, print_predictions=True),
    }
    plot_benchmarks(results, "./plots.png")
    plot_profiler_averages(results, "./profiler_plot.png")
