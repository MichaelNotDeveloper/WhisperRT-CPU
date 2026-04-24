import json
from dataclasses import dataclass, fields
from typing import Any, Literal
from audio_datasets import AudioTextDataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from utils import asr_metrics, ModuleTimer, plot_benchmarks, BenchmarkResult

import torch
import torch.nn as nn


# Code gets list of models, to be tested later on, methods as forward required to be implemented for each model,
# to be tested later on. Results are stored in json file, to be used for report generation.
@dataclass
class ModelsForBenchmark:
    BaseWhisper: nn.Module | None = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-base"
    )
    LargeWhisper: nn.Module | None = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-large-v3"
    )


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

        for field in fields(ModelsForBenchmark):
            print(field.name)
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
        original_encoder_forward = encoder_timer.wrap(model.model.encoder)
        original_decoder_forward = decoder_timer.wrap(model.model.decoder)

        for audio, text in self.dataset.take(sample_size):
            inputs = processor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
                return_attention_mask=True,
            )

            input_features = inputs.input_features.to(self.device)
            attention_mask = inputs.attention_mask.to(self.device)

            tokens = model.generate(
                input_features=input_features,
                attention_mask=attention_mask,
                language="ru",          # или "ru" для golos
                task="transcribe",
            )
            result = processor.batch_decode(tokens, skip_special_tokens=True)[0]

            metrics = asr_metrics(result, text)
            wer_hist.append(metrics["wer"])
            cer_hist.append(metrics["cer"])
            generated_texts.append(result)
            original_texts.append(text)

        print(encoder_timer.times)
        print(decoder_timer.times)
        
        return BenchmarkResult(
            wer_history=wer_hist,
            cer_history=cer_hist,
            generated_texts=generated_texts,
            original_texts=original_texts,
        )

    def get_models(self):
        return self.models

    def sample(data):
        pass


if __name__ == "__main__":
    results = Benchmark(
        dataset_name="golos",
        models={},
        device="CPU",
    ).run("BaseWhisper", sample_size=5)
    plot_benchmarks(results)