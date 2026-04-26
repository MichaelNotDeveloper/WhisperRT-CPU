from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Audio, load_dataset
from jiwer import cer, wer
from torch.optim import AdamW
from torch.utils.data import DataLoader, IterableDataset
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.modeling_whisper import shift_tokens_right

SAMPLE_RATE = 16_000

TRAIN100_FILES = [
    f"hf://datasets/openslr/librispeech_asr/clean/train.100/{i:04d}.parquet"
    for i in range(14)
]
TEST_CLEAN_FILES = [
    "hf://datasets/openslr/librispeech_asr/clean/test/0000.parquet",
]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--teacher-model", default="openai/whisper-large-v3-turbo")
    p.add_argument("--student-layers", type=int, nargs="+", default=[0, 3])
    p.add_argument("--output-dir", default="training/runs/distill_large_v3_turbo_p100_direct")

    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--language", default="english")
    p.add_argument("--task", default="transcribe")

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--train-batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)

    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--kl-weight", type=float, default=0.5)

    p.add_argument("--max-train-samples", type=int, default=500)
    p.add_argument("--max-eval-samples", type=int, default=32)
    p.add_argument("--max-latency-samples", type=int, default=8)
    p.add_argument("--eval-offset", type=int, default=None)
    p.add_argument("--latency-offset", type=int, default=None)
    p.add_argument("--shuffle-buffer", type=int, default=256)

    p.add_argument("--max-label-tokens", type=int, default=128)
    p.add_argument(
        "--max-gen-length",
        type=int,
        default=0,
        help="Total decoder length for eval/generation. 0 uses the model ceiling.",
    )
    p.add_argument("--max-new-tokens", type=int, default=0, help=argparse.SUPPRESS)

    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--plot-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--cpu-fallback", action="store_true")
    p.add_argument("--skip-final-teacher-eval", action="store_true")
    p.add_argument("--no-progress", action="store_true")

    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True


def get_dtype(name: str):
    return {"fp16": torch.float16, "fp32": torch.float32}[name]


def parse_supported_sms():
    sms = set()
    for arch in getattr(torch.cuda, "get_arch_list", lambda: [])():
        if not arch.startswith("sm_"):
            continue
        suffix = arch.split("_", 1)[1]
        if suffix.isdigit() and len(suffix) >= 2:
            sms.add((int(suffix[:-1]), int(suffix[-1])))
    return sms


def format_sms(sms):
    return ", ".join(f"sm_{major}{minor}" for major, minor in sorted(sms))


def resolve_runtime(args):
    device = torch.device(args.device)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this runtime.")

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        print(f"[runtime] GPU={gpu}, capability={cc[0]}.{cc[1]}")

        supported_sms = parse_supported_sms()
        if supported_sms and cc not in supported_sms:
            message = (
                f"Current PyTorch build does not include kernels for sm_{cc[0]}{cc[1]} ({gpu}). "
                f"Supported GPU architectures in this build: {format_sms(supported_sms)}. "
                "On Kaggle this means the current runtime cannot train on P100 with this PyTorch build. "
                "Switch the accelerator to T4/L4, install a PyTorch build with sm_60 support, "
                "or rerun with --cpu-fallback."
            )
            if args.cpu_fallback:
                print(f"[runtime] {message}")
                print("[runtime] switching to CPU fallback")
                args.device = "cpu"
                args.dtype = "fp32"
                return torch.device("cpu"), torch.float32
            raise RuntimeError(message)

        if cc[0] < 8 and args.dtype != "fp16":
            print("[runtime] pre-Ampere GPU detected, switching dtype to fp16")
            args.dtype = "fp16"

        if args.train_batch_size > 1:
            print("[runtime] memory warning: use --train-batch-size 1 if you get OOM")

    return device, get_dtype(args.dtype)


def autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype == torch.float16:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def progress_enabled(args):
    return not args.no_progress and sys.stderr.isatty()


def resolve_generation_length(model, processor, args):
    model_ceiling = getattr(model.config, "max_target_positions", None) or model.generation_config.max_length

    if args.max_gen_length and args.max_gen_length > 0:
        return min(args.max_gen_length, model_ceiling) if model_ceiling else args.max_gen_length

    if args.max_new_tokens and args.max_new_tokens > 0:
        prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        prompt_len = 1 + max((position for position, _ in prompt_ids), default=0)
        target_len = prompt_len + args.max_new_tokens
        return min(target_len, model_ceiling) if model_ceiling else target_len

    return model_ceiling


def normalize(text: str):
    return " ".join(str(text).strip().lower().split())


def load_parquet_stream(files, cache_dir: str | None):
    kwargs = {
        "data_files": files,
        "split": "train",
        "streaming": True,
    }
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    ds = load_dataset("parquet", **kwargs)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    keep = {"audio", "text"}
    drop = [c for c in ds.column_names if c not in keep]
    if drop:
        ds = ds.remove_columns(drop)

    return ds


def shuffle_stream(rows, buffer_size: int, seed: int):
    rng = random.Random(seed)
    buffer = []

    for row in rows:
        if len(buffer) < buffer_size:
            buffer.append(row)
            continue

        index = rng.randrange(len(buffer))
        yield buffer[index]
        buffer[index] = row

    rng.shuffle(buffer)
    for row in buffer:
        yield row


class StreamingAudioDataset(IterableDataset):
    def __init__(
        self,
        files,
        *,
        start: int = 0,
        limit: int | None = None,
        shuffle_buffer: int = 0,
        seed: int = 42,
        cache_dir: str | None = None,
    ):
        self.files = files
        self.start = start
        self.limit = limit
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.cache_dir = cache_dir

    def _rows(self):
        for file in self.files:
            ds = load_parquet_stream([file], self.cache_dir)
            for row in ds:
                yield row

    def __iter__(self):
        rows = self._rows()
        if self.shuffle_buffer > 1:
            rows = shuffle_stream(rows, self.shuffle_buffer, self.seed)

        seen = 0
        yielded = 0
        for row in rows:
            if seen < self.start:
                seen += 1
                continue
            if self.limit is not None and yielded >= self.limit:
                break
            seen += 1
            yielded += 1
            yield row


def load_train_dataset(args):
    ds = StreamingAudioDataset(
        TRAIN100_FILES,
        limit=args.max_train_samples,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    sources = ["openslr/librispeech_asr clean/train.100 parquet shards streamed lazily"]
    return ds, sources


def load_eval_dataset(limit: int, offset: int, args):
    return StreamingAudioDataset(
        TEST_CLEAN_FILES,
        start=offset,
        limit=limit,
        cache_dir=args.cache_dir,
    )


class BatchCollator:
    def __init__(self, processor: WhisperProcessor, max_label_tokens: int):
        self.processor = processor
        self.max_label_tokens = max_label_tokens
        self.bos = processor.tokenizer.bos_token_id
        self.max_audio_samples = processor.feature_extractor.n_samples

    def __call__(self, items):
        audios = [x["audio"]["array"] for x in items]
        texts = [normalize(x["text"]) for x in items]

        features = self.processor.feature_extractor(
            audios,
            sampling_rate=SAMPLE_RATE,
            padding="max_length",
            max_length=self.max_audio_samples,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        tokens = self.processor.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_label_tokens,
            return_tensors="pt",
        )

        labels = tokens.input_ids.masked_fill(tokens.attention_mask.ne(1), -100)

        if labels.size(1) > 0 and torch.all(labels[:, 0] == self.bos):
            labels = labels[:, 1:]

        return {
            "input_features": features.input_features,
            "attention_mask": getattr(features, "attention_mask", None),
            "labels": labels,
            "texts": texts,
            "seconds": [len(x["audio"]["array"]) / x["audio"]["sampling_rate"] for x in items],
        }


def make_loader(ds, batch_size: int, collator, pin_memory: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=pin_memory,
        persistent_workers=False,
    )


def validate_student_layers(teacher, indices: list[int]):
    n = len(teacher.model.decoder.layers)
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise ValueError(f"Invalid decoder layer indices {bad}; teacher decoder has {n} layers: 0..{n - 1}")


def build_models(args, device: torch.device, amp_dtype: torch.dtype):
    teacher = WhisperForConditionalGeneration.from_pretrained(args.teacher_model)
    validate_student_layers(teacher, args.student_layers)

    student = copy.deepcopy(teacher)
    old_layers = teacher.model.decoder.layers
    student.model.decoder.layers = nn.ModuleList([copy.deepcopy(old_layers[i]) for i in args.student_layers])
    student.config.decoder_layers = len(args.student_layers)
    student.model.decoder.config.decoder_layers = len(args.student_layers)
    for new_idx, layer in enumerate(student.model.decoder.layers):
        setattr(layer, "layer_idx", new_idx)
        if hasattr(layer, "self_attn"):
            setattr(layer.self_attn, "layer_idx", new_idx)
        if hasattr(layer, "encoder_attn"):
            setattr(layer.encoder_attn, "layer_idx", new_idx)

    teacher.config.use_cache = False
    student.config.use_cache = False

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    for p in student.parameters():
        p.requires_grad = False
    for p in student.model.decoder.parameters():
        p.requires_grad = True
    for p in student.proj_out.parameters():
        p.requires_grad = True

    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()

    dtype = amp_dtype if device.type == "cuda" else torch.float32
    teacher.to(device=device, dtype=dtype)
    student.to(device=device)
    student.model.decoder.to(dtype=torch.float32)
    student.proj_out.to(dtype=torch.float32)
    if device.type == "cuda" and amp_dtype == torch.float16:
        student.model.encoder.to(dtype=torch.float16)

    return teacher, student


def masked_kl(student_logits, teacher_logits, labels, temperature: float):
    mask = labels.ne(-100)
    if not mask.any():
        return student_logits.new_tensor(0.0)

    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.softmax(teacher_logits.float() / temperature, dim=-1)
    loss = F.kl_div(s, t, reduction="none").sum(dim=-1)

    return loss.masked_select(mask).mean() * temperature**2


def plot_history(history: dict[str, list[float]], out_path: Path):
    if not history["step"]:
        return

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax[0, 0].plot(history["step"], history["loss"], label="loss")
    ax[0, 0].plot(history["step"], history["ce"], label="ce", alpha=0.8)
    ax[0, 0].plot(history["step"], history["kl"], label="kl", alpha=0.8)
    ax[0, 0].set_title("Loss")
    ax[0, 0].legend()

    ax[0, 1].plot(history["step"], history["lr"], label="lr")
    ax[0, 1].set_title("LR")

    ax[1, 0].plot(history["eval_step"], history["eval_wer"], label="wer")
    ax[1, 0].plot(history["eval_step"], history["eval_cer"], label="cer")
    ax[1, 0].set_title("Eval")
    ax[1, 0].legend()

    ax[1, 1].plot(history["step"], history["samples_sec"], label="samples/s")
    ax[1, 1].set_title("Throughput")

    for axes in ax.flat:
        axes.grid(alpha=0.25)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_metrics(history: dict[str, list[float]], out_path: Path):
    out_path.write_text(json.dumps(history, indent=2, ensure_ascii=True), encoding="utf-8")


def move_batch(batch, device: torch.device):
    x = batch["input_features"].to(device, non_blocking=True)
    y = batch["labels"].to(device, non_blocking=True)

    m = batch["attention_mask"]
    if m is not None:
        m = m.to(device, non_blocking=True)

    return x, m, y


@torch.no_grad()
def evaluate(model, processor, loader, args, device: torch.device, amp_dtype: torch.dtype, desc: str, warmup: int = 0):
    model.eval()
    generation_config = copy.deepcopy(model.generation_config)
    generation_config.max_new_tokens = None
    generation_config.max_length = resolve_generation_length(model, processor, args)

    preds, refs = [], []
    wall, seconds, samples = 0.0, 0.0, 0

    for i, batch in enumerate(
        tqdm(loader, desc=desc, leave=False, disable=not progress_enabled(args), mininterval=5.0),
        start=1,
    ):
        x, m, _ = move_batch(batch, device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast_ctx(device, amp_dtype):
            ids = model.generate(
                input_features=x,
                attention_mask=m,
                generation_config=generation_config,
                language=args.language,
                task=args.task,
                num_beams=1,
                do_sample=False,
            )

        if device.type == "cuda":
            torch.cuda.synchronize()

        if i > warmup:
            wall += time.perf_counter() - t0
            decoded = processor.batch_decode(ids, skip_special_tokens=True)
            preds.extend(normalize(t) for t in decoded)
            refs.extend(batch["texts"])
            seconds += sum(batch["seconds"])
            samples += len(batch["texts"])

    return {
        "wer": float(wer(refs, preds)) if preds else float("nan"),
        "cer": float(cer(refs, preds)) if preds else float("nan"),
        "latency_ms": 1000.0 * wall / max(samples, 1),
        "rtf": wall / max(seconds, 1e-8),
        "samples": samples,
    }


def save_checkpoint(model, processor, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


def make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_grad_scaler(device: torch.device, amp_dtype: torch.dtype, params):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    enabled = enabled and all(param.dtype == torch.float32 for param in params if param.requires_grad)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train(teacher, student, processor, train_loader, eval_loader, args, out_dir: Path, device, amp_dtype):
    params = [p for p in student.parameters() if p.requires_grad]

    batches_per_epoch = math.ceil(args.max_train_samples / args.train_batch_size)
    total_steps = max(1, math.ceil(batches_per_epoch / args.grad_accum) * args.epochs)

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)
    scaler = build_grad_scaler(device, amp_dtype, params)

    history = {
        "step": [],
        "loss": [],
        "ce": [],
        "kl": [],
        "lr": [],
        "samples_sec": [],
        "eval_step": [],
        "eval_wer": [],
        "eval_cer": [],
    }

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    pbar = tqdm(
        total=total_steps,
        desc="train",
        disable=not progress_enabled(args),
        mininterval=5.0,
    )

    for epoch in range(args.epochs):
        student.train()
        max_micro_steps = math.ceil(args.max_train_samples / args.train_batch_size)
        accum_start = None
        accum_samples = 0

        for micro_step, batch in enumerate(train_loader, start=1):
            if micro_step > max_micro_steps:
                break

            if accum_start is None:
                accum_start = time.perf_counter()

            x, m, y = move_batch(batch, device)
            accum_samples += x.size(0)
            dec = shift_tokens_right(y, student.config.pad_token_id, student.config.decoder_start_token_id)
            dec_mask = dec.ne(student.config.pad_token_id).long()

            window_start = ((micro_step - 1) // args.grad_accum) * args.grad_accum + 1
            window_end = min(window_start + args.grad_accum - 1, max_micro_steps)
            microbatches_in_update = window_end - window_start + 1

            with autocast_ctx(device, amp_dtype):
                with torch.no_grad():
                    t_logits = teacher(
                        input_features=x,
                        attention_mask=m,
                        decoder_input_ids=dec,
                        decoder_attention_mask=dec_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits

                s_logits = student(
                    input_features=x,
                    attention_mask=m,
                    decoder_input_ids=dec,
                    decoder_attention_mask=dec_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits

                ce_loss = F.cross_entropy(
                    s_logits.reshape(-1, s_logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                )
                kl_loss = masked_kl(s_logits, t_logits, y, args.temperature)
                loss = args.ce_weight * ce_loss + args.kl_weight * kl_loss

            scaled = loss / microbatches_in_update
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            is_update_step = micro_step == window_end
            if not is_update_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            pbar.update(1)

            elapsed = time.perf_counter() - accum_start
            history["step"].append(global_step)
            history["loss"].append(float(loss.item()))
            history["ce"].append(float(ce_loss.item()))
            history["kl"].append(float(kl_loss.item()))
            history["lr"].append(float(scheduler.get_last_lr()[0]))
            history["samples_sec"].append(accum_samples / max(elapsed, 1e-8))
            accum_start = None
            accum_samples = 0

            if global_step % args.eval_every == 0:
                metrics = evaluate(student, processor, eval_loader, args, device, amp_dtype, f"eval@{global_step}")
                history["eval_step"].append(global_step)
                history["eval_wer"].append(metrics["wer"])
                history["eval_cer"].append(metrics["cer"])
                student.train()
                print(
                    f"[step {global_step}] loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} wer={metrics['wer']:.4f} grad={float(grad_norm):.3f}"
                )
            elif global_step % max(1, args.plot_every) == 0:
                print(
                    f"[step {global_step}] loss={loss.item():.4f} ce={ce_loss.item():.4f} "
                    f"kl={kl_loss.item():.4f} grad={float(grad_norm):.3f}"
                )

            if global_step % max(1, args.plot_every) == 0:
                plot_history(history, out_dir / "live.png")
                save_metrics(history, out_dir / "history.json")

            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(student, processor, out_dir / f"checkpoint-{global_step}")

        print(f"[epoch {epoch + 1}/{args.epochs}] done")

    pbar.close()
    plot_history(history, out_dir / "live.png")
    save_metrics(history, out_dir / "history.json")
    save_checkpoint(student, processor, out_dir / "final_student")

    return history


def main():
    args = parse_args()
    device, amp_dtype = resolve_runtime(args)

    if args.eval_offset is None:
        args.eval_offset = 0
    if args.latency_offset is None:
        args.latency_offset = args.max_eval_samples

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=True), encoding="utf-8")

    processor = WhisperProcessor.from_pretrained(args.teacher_model)
    processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)
    collator = BatchCollator(processor, args.max_label_tokens)

    train_ds, train_sources = load_train_dataset(args)
    eval_ds = load_eval_dataset(args.max_eval_samples, args.eval_offset, args)
    latency_ds = load_eval_dataset(args.max_latency_samples, args.latency_offset, args)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_ds, args.train_batch_size, collator, pin_memory)
    eval_loader = make_loader(eval_ds, args.eval_batch_size, collator, pin_memory)
    latency_loader = make_loader(latency_ds, 1, collator, pin_memory)

    teacher, student = build_models(args, device, amp_dtype)

    print("teacher:", args.teacher_model)
    print("student layers:", args.student_layers)
    print("dtype:", args.dtype)
    print("recipe: direct parquet streaming, freeze encoder, train decoder + proj_out, loss = CE + KL")
    print("train sources:")
    for src in train_sources:
        print("  -", src)

    train(teacher, student, processor, train_loader, eval_loader, args, out_dir, device, amp_dtype)

    if args.skip_final_teacher_eval:
        teacher_eval = None
        teacher_lat = None
    else:
        teacher_eval = evaluate(teacher, processor, eval_loader, args, device, amp_dtype, "teacher-eval")
        teacher_lat = evaluate(teacher, processor, latency_loader, args, device, amp_dtype, "teacher-lat", warmup=2)

    student_eval = evaluate(student, processor, eval_loader, args, device, amp_dtype, "student-eval")
    student_lat = evaluate(student, processor, latency_loader, args, device, amp_dtype, "student-lat", warmup=2)

    summary = {
        "teacher_eval": teacher_eval,
        "student_eval": student_eval,
        "teacher_latency": teacher_lat,
        "student_latency": student_lat,
        "student_layers": args.student_layers,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    if teacher_eval is not None and teacher_lat is not None:
        print(
            f"teacher | WER={teacher_eval['wer']:.4f} CER={teacher_eval['cer']:.4f} "
            f"latency={teacher_lat['latency_ms']:.2f}ms RTF={teacher_lat['rtf']:.4f}"
        )

    print(
        f"student | WER={student_eval['wer']:.4f} CER={student_eval['cer']:.4f} "
        f"latency={student_lat['latency_ms']:.2f}ms RTF={student_lat['rtf']:.4f}"
    )
    print("artifacts:", out_dir.resolve())


if __name__ == "__main__":
    main()
