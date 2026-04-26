from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer, wer
from tqdm.auto import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.modeling_whisper import _expand_mask, _make_causal_mask, shift_tokens_right

from distill_large_v3_turbo_a100 import (
    BatchCollator,
    autocast_ctx,
    build_grad_scaler,
    load_eval_dataset,
    load_train_dataset,
    make_loader,
    make_scheduler,
    move_batch,
    normalize,
    progress_enabled,
    resolve_runtime,
    save_metrics,
    seed_everything,
)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model-name", default="openai/whisper-large-v3-turbo")
    p.add_argument("--future-tokens", type=int, default=3)
    p.add_argument("--output-dir", default="training/runs/mtp_large_v3_turbo")

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

    p.add_argument("--base-ce-weight", type=float, default=1.0)
    p.add_argument("--mtp-ce-weight", type=float, default=2.0)

    p.add_argument("--max-train-samples", type=int, default=500)
    p.add_argument("--max-eval-samples", type=int, default=32)
    p.add_argument("--max-latency-samples", type=int, default=8)
    p.add_argument("--eval-offset", type=int, default=None)
    p.add_argument("--latency-offset", type=int, default=None)
    p.add_argument("--shuffle-buffer", type=int, default=256)

    p.add_argument("--max-label-tokens", type=int, default=128)
    p.add_argument("--max-gen-length", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=0, help=argparse.SUPPRESS)

    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--plot-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache-dir", default=None)

    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--cpu-fallback", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-final-latency", action="store_true")
    p.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true")
    p.add_argument("--no-freeze-encoder", dest="freeze_encoder", action="store_false")
    p.set_defaults(freeze_encoder=True)

    return p.parse_args()


def resolve_generation_length(model, processor, args):
    model_ceiling = getattr(model.base.config, "max_target_positions", None) or model.base.generation_config.max_length

    if args.max_gen_length and args.max_gen_length > 0:
        return min(args.max_gen_length, model_ceiling) if model_ceiling else args.max_gen_length

    if args.max_new_tokens and args.max_new_tokens > 0:
        prompt_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        prompt_len = 1 + max((position for position, _ in prompt_ids), default=0)
        target_len = prompt_len + args.max_new_tokens
        return min(target_len, model_ceiling) if model_ceiling else target_len

    return model_ceiling


def masked_ce(logits, labels):
    mask = labels.ne(-100)
    if not mask.any():
        return logits.new_tensor(0.0)
    return F.cross_entropy(logits[mask], labels[mask])


def accuracy_counts(logits, labels):
    mask = labels.ne(-100)
    if not mask.any():
        return 0, 0
    pred = logits.argmax(dim=-1)
    correct = pred.eq(labels).masked_select(mask).sum().item()
    total = mask.sum().item()
    return int(correct), int(total)


def format_block_accs(block_accs):
    return " ".join(f"b{i}={acc:.4f}" for i, acc in enumerate(block_accs))


class OneLayerDecoderHead(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.layer = copy.deepcopy(decoder.layers[-1])
        self.layer_norm = copy.deepcopy(decoder.layer_norm)
        layer = self.layer
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = 0
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = 0
        if hasattr(layer, "encoder_attn"):
            layer.encoder_attn.layer_idx = 0

    def forward(self, hidden_states, attention_mask, encoder_hidden_states):
        input_shape = hidden_states.shape[:2]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                hidden_states.dtype,
                device=hidden_states.device,
                past_key_values_length=0,
            )
        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(attention_mask, hidden_states.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )
        hidden_states = self.layer(
            hidden_states,
            attention_mask=combined_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            output_attentions=False,
        )[0]
        return self.layer_norm(hidden_states)


class WhisperMTP(nn.Module):
    def __init__(self, model_name: str, future_tokens: int, freeze_encoder: bool, gradient_checkpointing: bool):
        super().__init__()
        self.base = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.base.config.use_cache = False
        self.future_tokens = future_tokens
        self.mtp_heads = nn.ModuleList(
            [OneLayerDecoderHead(self.base.model.decoder) for _ in range(future_tokens)]
        )

        if gradient_checkpointing:
            self.base.gradient_checkpointing_enable()

        for p in self.base.parameters():
            p.requires_grad = False
        if not freeze_encoder:
            for p in self.base.model.encoder.parameters():
                p.requires_grad = True
        for p in self.base.model.decoder.parameters():
            p.requires_grad = True
        for p in self.base.proj_out.parameters():
            p.requires_grad = True
        for p in self.mtp_heads.parameters():
            p.requires_grad = True

    @property
    def config(self):
        return self.base.config

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def prepare_for_runtime(self, device: torch.device, amp_dtype: torch.dtype, freeze_encoder: bool):
        self.to(device=device)
        self.base.model.decoder.to(dtype=torch.float32)
        self.base.proj_out.to(dtype=torch.float32)
        self.mtp_heads.to(dtype=torch.float32)
        if freeze_encoder and device.type == "cuda" and amp_dtype == torch.float16:
            self.base.model.encoder.to(dtype=torch.float16)

    def encode(self, input_features, attention_mask=None):
        return self.base.model.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

    def make_targets(self, labels):
        targets = []
        for shift in range(self.future_tokens + 1):
            target = labels.new_full(labels.shape, -100)
            if shift == 0:
                target.copy_(labels)
            elif labels.size(1) > shift:
                target[:, :-shift] = labels[:, shift:]
            targets.append(target)
        return targets

    def forward(self, input_features, attention_mask, decoder_input_ids, decoder_attention_mask):
        encoder_hidden_states = self.encode(input_features, attention_mask)
        decoder_hidden_states = self.base.model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        ).last_hidden_state

        logits = [self.base.proj_out(decoder_hidden_states)]

        for head in self.mtp_heads:
            head_hidden_states = head(decoder_hidden_states, decoder_attention_mask, encoder_hidden_states)
            logits.append(self.base.proj_out(head_hidden_states))

        return logits

    def prompt_ids(self, processor, language: str, task: str):
        return [self.base.config.decoder_start_token_id] + [
            token_id for _, token_id in sorted(processor.get_decoder_prompt_ids(language=language, task=task))
        ]

    @torch.no_grad()
    def generate(
        self,
        input_features,
        attention_mask,
        processor,
        language: str,
        task: str,
        max_length: int,
        return_stats: bool = False,
    ):
        if input_features.size(0) != 1:
            raise ValueError("WhisperMTP.generate currently expects batch_size=1.")

        encoder_hidden_states = self.encode(input_features, attention_mask)
        ids = torch.tensor([self.prompt_ids(processor, language, task)], device=input_features.device)
        stats = {"accepted": 0, "proposed": 0, "steps": 0}

        while ids.size(1) < max_length:
            decoder_attention_mask = torch.ones_like(ids)
            decoder_hidden_states = self.base.model.decoder(
                input_ids=ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=True,
            ).last_hidden_state

            block = [self.base.proj_out(decoder_hidden_states[:, -1, :]).argmax(dim=-1, keepdim=True)]

            for head in self.mtp_heads:
                head_hidden_states = head(decoder_hidden_states, decoder_attention_mask, encoder_hidden_states)
                block.append(self.base.proj_out(head_hidden_states[:, -1, :]).argmax(dim=-1, keepdim=True))

            accepted = [block[0]]
            verify_ids = torch.cat([ids, block[0]], dim=-1)
            stats["proposed"] += len(block)
            stats["steps"] += 1
            for speculative in block[1:]:
                if verify_ids.size(1) >= max_length:
                    break
                verify_hidden = self.base.model.decoder(
                    input_ids=verify_ids,
                    attention_mask=torch.ones_like(verify_ids),
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=True,
                ).last_hidden_state
                verified = self.base.proj_out(verify_hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
                if not torch.equal(verified, speculative):
                    break
                accepted.append(speculative)
                verify_ids = torch.cat([verify_ids, speculative], dim=-1)

            for token in accepted:
                if ids.size(1) >= max_length:
                    break
                ids = torch.cat([ids, token], dim=-1)
                stats["accepted"] += 1
                if token.item() == self.base.generation_config.eos_token_id:
                    return (ids, stats) if return_stats else ids

        return (ids, stats) if return_stats else ids


def build_model(args, device: torch.device, amp_dtype: torch.dtype):
    model = WhisperMTP(
        model_name=args.model_name,
        future_tokens=args.future_tokens,
        freeze_encoder=args.freeze_encoder,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.prepare_for_runtime(device, amp_dtype, args.freeze_encoder)
    return model


def mtp_losses(model, logits_by_horizon, labels):
    targets = model.make_targets(labels)
    losses = [masked_ce(logits, target) for logits, target in zip(logits_by_horizon, targets)]
    base_ce = losses[0]
    mtp_ce = torch.stack(losses[1:]).mean() if len(losses) > 1 else base_ce.new_tensor(0.0)
    return base_ce, mtp_ce


def plot_history(history: dict[str, list[float]], out_path: Path):
    if not history["step"]:
        return

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)

    ax[0, 0].plot(history["step"], history["loss"], label="loss")
    ax[0, 0].plot(history["step"], history["base_ce"], label="base_ce", alpha=0.8)
    ax[0, 0].plot(history["step"], history["mtp_ce"], label="mtp_ce", alpha=0.8)
    ax[0, 0].set_title("Loss")
    ax[0, 0].legend()

    ax[0, 1].plot(history["step"], history["lr"], label="lr")
    ax[0, 1].set_title("LR")

    ax[0, 2].plot(history["step"], history["samples_sec"], label="samples/s")
    ax[0, 2].set_title("Throughput")

    ax[1, 0].plot(history["eval_step"], history["eval_wer"], label="wer")
    ax[1, 0].plot(history["eval_step"], history["eval_cer"], label="cer")
    ax[1, 0].set_title("Eval Text")
    ax[1, 0].legend()

    ax[1, 1].plot(history["eval_step"], history["eval_base_acc"], label="base_acc")
    ax[1, 1].plot(history["eval_step"], history["eval_mtp_acc"], label="mtp_acc")
    ax[1, 1].set_title("Eval Accuracy")
    ax[1, 1].legend()

    for key in sorted(k for k in history if k.startswith("eval_block_acc_")):
        block_idx = key.rsplit("_", 1)[-1]
        ax[1, 2].plot(history["eval_step"], history[key], label=f"block_{block_idx}")
    ax[1, 2].set_title("Block Accuracy")
    ax[1, 2].legend()

    for axes in ax.flat:
        axes.grid(alpha=0.25)

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_checkpoint(model, processor, args, path: Path):
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": args.model_name,
            "future_tokens": args.future_tokens,
            "freeze_encoder": args.freeze_encoder,
        },
        path / "mtp.pt",
    )
    processor.save_pretrained(path)


@torch.no_grad()
def evaluate(model, processor, loader, args, device: torch.device, amp_dtype: torch.dtype, desc: str, warmup: int = 0):
    model.eval()
    preds, refs = [], []
    wall, seconds, samples = 0.0, 0.0, 0
    accepted, proposed = 0, 0
    base_correct, base_total = 0, 0
    mtp_correct, mtp_total = 0, 0
    block_correct = [0 for _ in range(model.future_tokens + 1)]
    block_total = [0 for _ in range(model.future_tokens + 1)]
    max_length = resolve_generation_length(model, processor, args)

    for i, batch in enumerate(
        tqdm(loader, desc=desc, leave=False, disable=not progress_enabled(args), mininterval=5.0),
        start=1,
    ):
        x, m, y = move_batch(batch, device)
        dec = shift_tokens_right(y, model.config.pad_token_id, model.config.decoder_start_token_id)
        dec_mask = dec.ne(model.config.pad_token_id).long()

        with autocast_ctx(device, amp_dtype):
            logits_by_horizon = model(
                input_features=x,
                attention_mask=m,
                decoder_input_ids=dec,
                decoder_attention_mask=dec_mask,
            )
        targets = model.make_targets(y)
        correct, total = accuracy_counts(logits_by_horizon[0], targets[0])
        base_correct += correct
        base_total += total
        block_correct[0] += correct
        block_total[0] += total
        for block_idx, (logits, target) in enumerate(zip(logits_by_horizon[1:], targets[1:]), start=1):
            correct, total = accuracy_counts(logits, target)
            mtp_correct += correct
            mtp_total += total
            block_correct[block_idx] += correct
            block_total[block_idx] += total

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with autocast_ctx(device, amp_dtype):
            ids, stats = model.generate(
                input_features=x,
                attention_mask=m,
                processor=processor,
                language=args.language,
                task=args.task,
                max_length=max_length,
                return_stats=True,
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
            accepted += stats["accepted"]
            proposed += stats["proposed"]

    return {
        "wer": float(wer(refs, preds)) if preds else float("nan"),
        "cer": float(cer(refs, preds)) if preds else float("nan"),
        "base_acc": base_correct / max(base_total, 1),
        "mtp_acc": mtp_correct / max(mtp_total, 1),
        "block_accs": [correct / max(total, 1) for correct, total in zip(block_correct, block_total)],
        "latency_ms": 1000.0 * wall / max(samples, 1),
        "rtf": wall / max(seconds, 1e-8),
        "accept_ratio": accepted / max(proposed, 1),
        "samples": samples,
    }


def train(model, processor, train_loader, eval_loader, args, out_dir: Path, device, amp_dtype):
    params = model.trainable_parameters()
    batches_per_epoch = math.ceil(args.max_train_samples / args.train_batch_size)
    total_steps = max(1, math.ceil(batches_per_epoch / args.grad_accum) * args.epochs)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    scheduler = make_scheduler(optimizer, total_steps, args.warmup_ratio)
    scaler = build_grad_scaler(device, amp_dtype, params)

    history = {
        "step": [],
        "loss": [],
        "base_ce": [],
        "mtp_ce": [],
        "lr": [],
        "samples_sec": [],
        "eval_step": [],
        "eval_wer": [],
        "eval_cer": [],
        "eval_base_acc": [],
        "eval_mtp_acc": [],
    }
    for block_idx in range(model.future_tokens + 1):
        history[f"eval_block_acc_{block_idx}"] = []

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    pbar = tqdm(total=total_steps, desc="train", disable=not progress_enabled(args), mininterval=5.0)

    for epoch in range(args.epochs):
        model.train()
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
            dec = shift_tokens_right(y, model.config.pad_token_id, model.config.decoder_start_token_id)
            dec_mask = dec.ne(model.config.pad_token_id).long()

            window_start = ((micro_step - 1) // args.grad_accum) * args.grad_accum + 1
            window_end = min(window_start + args.grad_accum - 1, max_micro_steps)
            microbatches_in_update = window_end - window_start + 1

            with autocast_ctx(device, amp_dtype):
                logits_by_horizon = model(
                    input_features=x,
                    attention_mask=m,
                    decoder_input_ids=dec,
                    decoder_attention_mask=dec_mask,
                )
                base_ce, mtp_ce = mtp_losses(model, logits_by_horizon, y)
                loss = args.base_ce_weight * base_ce + args.mtp_ce_weight * mtp_ce

            scaled = loss / microbatches_in_update
            if scaler.is_enabled():
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            if micro_step != window_end:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)

            if scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                did_step = scaler.get_scale() >= old_scale
            else:
                optimizer.step()
                did_step = True

            optimizer.zero_grad(set_to_none=True)
            if not did_step:
                accum_start = None
                accum_samples = 0
                continue

            scheduler.step()
            global_step += 1
            pbar.update(1)

            elapsed = time.perf_counter() - accum_start
            history["step"].append(global_step)
            history["loss"].append(float(loss.item()))
            history["base_ce"].append(float(base_ce.item()))
            history["mtp_ce"].append(float(mtp_ce.item()))
            history["lr"].append(float(scheduler.get_last_lr()[0]))
            history["samples_sec"].append(accum_samples / max(elapsed, 1e-8))
            accum_start = None
            accum_samples = 0

            if global_step % args.eval_every == 0:
                metrics = evaluate(model, processor, eval_loader, args, device, amp_dtype, f"eval@{global_step}")
                history["eval_step"].append(global_step)
                history["eval_wer"].append(metrics["wer"])
                history["eval_cer"].append(metrics["cer"])
                history["eval_base_acc"].append(metrics["base_acc"])
                history["eval_mtp_acc"].append(metrics["mtp_acc"])
                for block_idx, acc in enumerate(metrics["block_accs"]):
                    history[f"eval_block_acc_{block_idx}"].append(acc)
                model.train()
                print(
                    f"[step {global_step}] loss={loss.item():.4f} base_ce={base_ce.item():.4f} "
                    f"mtp_ce={mtp_ce.item():.4f} wer={metrics['wer']:.4f} "
                    f"base_acc={metrics['base_acc']:.4f} mtp_acc={metrics['mtp_acc']:.4f} "
                    f"{format_block_accs(metrics['block_accs'])} "
                    f"grad={float(grad_norm):.3f}"
                )
            elif global_step % max(1, args.plot_every) == 0:
                print(
                    f"[step {global_step}] loss={loss.item():.4f} base_ce={base_ce.item():.4f} "
                    f"mtp_ce={mtp_ce.item():.4f} grad={float(grad_norm):.3f}"
                )

            if global_step % max(1, args.plot_every) == 0:
                plot_history(history, out_dir / "live.png")
                save_metrics(history, out_dir / "history.json")

            if args.save_every and global_step % args.save_every == 0:
                save_checkpoint(model, processor, args, out_dir / f"checkpoint-{global_step}")

        print(f"[epoch {epoch + 1}/{args.epochs}] done")

    pbar.close()
    plot_history(history, out_dir / "live.png")
    save_metrics(history, out_dir / "history.json")
    save_checkpoint(model, processor, args, out_dir / "final_model")

    return history


def main():
    args = parse_args()
    device, amp_dtype = resolve_runtime(args)

    if args.eval_batch_size != 1:
        print("[runtime] WhisperMTP eval uses speculative decoding; forcing --eval-batch-size 1")
        args.eval_batch_size = 1
    if args.eval_offset is None:
        args.eval_offset = 0
    if args.latency_offset is None:
        args.latency_offset = args.max_eval_samples

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    (out_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=True), encoding="utf-8")

    processor = WhisperProcessor.from_pretrained(args.model_name)
    processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)
    collator = BatchCollator(processor, args.max_label_tokens)

    train_ds, train_sources = load_train_dataset(args)
    eval_ds = load_eval_dataset(args.max_eval_samples, args.eval_offset, args)
    latency_ds = load_eval_dataset(args.max_latency_samples, args.latency_offset, args)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(train_ds, args.train_batch_size, collator, pin_memory)
    eval_loader = make_loader(eval_ds, args.eval_batch_size, collator, pin_memory)
    latency_loader = make_loader(latency_ds, 1, collator, pin_memory)

    model = build_model(args, device, amp_dtype)

    print("model:", args.model_name)
    print("future tokens:", args.future_tokens)
    print("dtype:", args.dtype)
    print("recipe: direct parquet streaming, freeze encoder, train decoder + proj_out + mtp heads")
    print("train sources:")
    for src in train_sources:
        print("  -", src)

    train(model, processor, train_loader, eval_loader, args, out_dir, device, amp_dtype)

    student_eval = evaluate(model, processor, eval_loader, args, device, amp_dtype, "mtp-eval")
    student_lat = None
    if not args.skip_final_latency:
        student_lat = evaluate(model, processor, latency_loader, args, device, amp_dtype, "mtp-lat", warmup=2)

    summary = {
        "student_eval": student_eval,
        "student_latency": student_lat,
        "future_tokens": args.future_tokens,
        "freeze_encoder": args.freeze_encoder,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    if student_lat is None:
        print(
            f"mtp | WER={student_eval['wer']:.4f} CER={student_eval['cer']:.4f} "
            f"base_acc={student_eval['base_acc']:.4f} mtp_acc={student_eval['mtp_acc']:.4f} "
            f"{format_block_accs(student_eval['block_accs'])}"
        )
    else:
        print(
            f"mtp | WER={student_eval['wer']:.4f} CER={student_eval['cer']:.4f} "
            f"base_acc={student_eval['base_acc']:.4f} mtp_acc={student_eval['mtp_acc']:.4f} "
            f"{format_block_accs(student_eval['block_accs'])} "
            f"latency={student_lat['latency_ms']:.2f}ms RTF={student_lat['rtf']:.4f} "
            f"accept={student_lat['accept_ratio']:.3f}"
        )
    print("artifacts:", out_dir.resolve())


if __name__ == "__main__":
    main()
