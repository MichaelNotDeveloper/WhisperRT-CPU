import copy
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from audio_datasets import AudioTextDataset
from baseline_models import PrunedTurboDecoder
from utils import BenchmarkResult, ModuleTimer, asr_metrics, plot_benchmarks

SAMPLE_SIZE = 256
MODEL_NAME = "openai/whisper-large-v3-turbo"
PRUNED_CHECKPOINT = "2DecoderModelWeights"


def _build_base_model() -> WhisperForConditionalGeneration:
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )
    model.generation_config.do_sample = False
    model.generation_config.return_timestamps = False
    model.generation_config.use_cache = True
    model.eval()
    return model


def _build_quantized_model(
    base_model: WhisperForConditionalGeneration,
    dtype: torch.dtype,
) -> WhisperForConditionalGeneration:
    model = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        {nn.Linear},
        dtype=dtype,
    )
    model.eval()
    return model


class Benchmark:
    def __init__(
        self,
        dataset_name: str,
        device: str = "cpu",
    ):
        if device != "cpu":
            raise ValueError("quantized_models.py supports CPU benchmark only")

        self.device = "cpu"
        self.dataset = AudioTextDataset(dataset_name)
        self._warmed_up = set()
        base_processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        pruned_processor = WhisperProcessor.from_pretrained(PRUNED_CHECKPOINT)
        self.processors = {
            "Baseline": base_processor,
            "DynamicInt8": base_processor,
            "DynamicFp16": base_processor,
            "Pruned": pruned_processor,
            "PrunedDynamicInt8": pruned_processor,
            "PrunedDynamicFp16": pruned_processor,
        }

        base_model = _build_base_model()
        pruned_model = PrunedTurboDecoder(PRUNED_CHECKPOINT)()
        self.models = {
            "Baseline": base_model,
            "DynamicInt8": _build_quantized_model(base_model, torch.qint8),
            "DynamicFp16": _build_quantized_model(base_model, torch.float16),
            "Pruned": pruned_model,
            "PrunedDynamicInt8": _build_quantized_model(pruned_model, torch.qint8),
            "PrunedDynamicFp16": _build_quantized_model(pruned_model, torch.float16),
        }

    def warmup(self, model_name: str) -> None:
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
                    language="en",
                    task="transcribe",
                )
        finally:
            model.model.encoder.forward = original_encoder
            model.model.decoder.forward = original_decoder

        self._warmed_up.add(model_name)

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
        encoder_speed = []
        decoder_speed = []
        processor_speed = []

        self.warmup(model_name)

        encoder_timer = ModuleTimer()
        decoder_timer = ModuleTimer()
        original_encoder = encoder_timer.wrap(model.model.encoder)
        original_decoder = decoder_timer.wrap(model.model.decoder)

        try:
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
                    encoder_start = len(encoder_timer.times)
                    decoder_start = len(decoder_timer.times)

                    tokens = model.generate(
                        input_features=inputs.input_features.to(self.device),
                        attention_mask=inputs.attention_mask.to(self.device),
                        language="en",
                        task="transcribe",
                    )
                    encoder_speed.append(sum(encoder_timer.times[encoder_start:]))
                    decoder_speed.append(sum(decoder_timer.times[decoder_start:]))
                    result = processor.batch_decode(tokens, skip_special_tokens=True)[0]

                    sample_runtime = time.perf_counter() - sample_start
                    audio_duration = len(audio["array"]) / audio["sampling_rate"]
                    audio_time_ratio.append(sample_runtime / audio_duration)

                    if print_predictions:
                        tqdm.write(f"[{model_name} #{sample_index}] target: {text}")
                        tqdm.write(f"[{model_name} #{sample_index}] prediction: {result}")

                    metrics = asr_metrics(result, text)
                    wer_hist.append(metrics["wer"])
                    cer_hist.append(metrics["cer"])
                    generated_texts.append(result)
                    original_texts.append(text)
        finally:
            model.model.encoder.forward = original_encoder
            model.model.decoder.forward = original_decoder


        return BenchmarkResult(
            wer_history=wer_hist,
            cer_history=cer_hist,
            generated_texts=generated_texts,
            original_texts=original_texts,
            audio_time_ratio=audio_time_ratio,
            encoder_speed=encoder_speed,
            decoder_speed=decoder_speed,
            processor_speed=processor_speed,
        )


if __name__ == "__main__":
    bench = Benchmark(
        dataset_name="earnings22",
    )

    baseline_result = bench.run("Baseline", sample_size=SAMPLE_SIZE)
    pruned_result = bench.run("Pruned", sample_size=SAMPLE_SIZE)

    full_results = {
        "Baseline": baseline_result,
        "DynamicInt8": bench.run("DynamicInt8", sample_size=SAMPLE_SIZE),
        "DynamicFp16": bench.run("DynamicFp16", sample_size=SAMPLE_SIZE),
    }
    pruned_results = {
        "Baseline": baseline_result,
        "Pruned": pruned_result,
        "PrunedDynamicInt8": bench.run("PrunedDynamicInt8", sample_size=SAMPLE_SIZE),
        "PrunedDynamicFp16": bench.run("PrunedDynamicFp16", sample_size=SAMPLE_SIZE),
    }

    plot_benchmarks(full_results, "./quantization_plots.png")
    plot_benchmarks(pruned_results, "./pruned_quantization_plots.png")
