import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

DEFAULT_MTP_CHECKPOINT = Path(__file__).resolve().parents[1] / "mtp_weights"


class BaselineSmall:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(
            dtype=torch.float32
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


class BaselineTurbo:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo"
        ).to(dtype=torch.float32)
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


class BaselineLarge:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(
            dtype=torch.float32
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


class BaselineTurboParams:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v3-turbo"
        ).to(dtype=torch.float32)
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


class TorchCompileTurboParams:
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
        dtype: torch.dtype = torch.float32,
        compile_mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool | None = None,
    ):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

        torch._logging.set_logs(graph_code=True)

        self.model.model.encoder = torch.compile(
            self.model.model.encoder,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

        self.model.model.decoder = torch.compile(
            self.model.model.decoder,
            mode=compile_mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


class PrunedTurboDecoder:
    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "openai/whisper-large-v3-turbo",
        dtype: torch.dtype = torch.float32,
        strict: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model = self._load_model(
            checkpoint_path=self.checkpoint_path,
            model_name=model_name,
            dtype=dtype,
            strict=strict,
        )
        self.model.generation_config.do_sample = False
        self.model.generation_config.return_timestamps = False
        self.model.generation_config.use_cache = True

    def _load_model(
        self,
        checkpoint_path: Path,
        model_name: str,
        dtype: torch.dtype,
        strict: bool,
    ) -> WhisperForConditionalGeneration:
        if checkpoint_path.is_dir():
            try:
                return WhisperForConditionalGeneration.from_pretrained(
                    str(checkpoint_path),
                    torch_dtype=dtype,
                )
            except Exception:
                pass

        state_dict = self._load_state_dict(checkpoint_path)
        decoder_layers = self._infer_decoder_layers(state_dict)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self._shrink_decoder(model, decoder_layers)
        self.load_info = model.load_state_dict(state_dict, strict=strict)
        return model.to(dtype=dtype)

    def _load_state_dict(self, checkpoint_path: Path) -> dict[str, torch.Tensor]:
        if checkpoint_path.is_dir():
            checkpoint_path = self._resolve_weight_file(checkpoint_path)

        if checkpoint_path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state_dict = load_file(str(checkpoint_path))
        else:
            state_dict = torch.load(checkpoint_path, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            maybe_state_dict = state_dict["state_dict"]
            if isinstance(maybe_state_dict, dict):
                state_dict = maybe_state_dict

        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint payload type: {type(state_dict)!r}")

        return state_dict

    def _resolve_weight_file(self, checkpoint_dir: Path) -> Path:
        for name in (
            "model.safetensors",
            "modeldecode.safetensors",
            "pytorch_model.bin",
        ):
            candidate = checkpoint_dir / name
            if candidate.exists():
                return candidate

        matches = sorted(
            path
            for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix in {".safetensors", ".bin"}
        )
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Could not find weights inside {checkpoint_dir}. Expected a .safetensors or .bin file."
        )

    def _infer_decoder_layers(self, state_dict: dict[str, torch.Tensor]) -> int:
        prefix = "model.decoder.layers."
        indices = {
            int(key[len(prefix) :].split(".", 1)[0]) for key in state_dict if key.startswith(prefix)
        }
        if not indices:
            raise ValueError("Could not infer decoder layer count from checkpoint state_dict.")
        return max(indices) + 1

    def _shrink_decoder(
        self,
        model: WhisperForConditionalGeneration,
        decoder_layers: int,
    ) -> None:
        old_layers = model.model.decoder.layers
        if decoder_layers > len(old_layers):
            raise ValueError(
                f"Checkpoint expects {decoder_layers} decoder layers, "
                f"but base model has only {len(old_layers)}."
            )

        model.model.decoder.layers = nn.ModuleList(
            [copy.deepcopy(old_layers[i]) for i in range(decoder_layers)]
        )
        model.config.decoder_layers = decoder_layers
        model.model.decoder.config.decoder_layers = decoder_layers
        for new_idx, layer in enumerate(model.model.decoder.layers):
            layer.layer_idx = new_idx
            if hasattr(layer, "self_attn"):
                layer.self_attn.layer_idx = new_idx
            if hasattr(layer, "encoder_attn"):
                layer.encoder_attn.layer_idx = new_idx

    def __call__(self):
        return self.model

    def get_timer(self):
        return None


def _make_causal_mask(
    input_shape: tuple[int, int],
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
) -> torch.Tensor:
    _, tgt_len = input_shape
    mask = torch.triu(
        torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device),
        diagonal=1,
    )
    if past_key_values_length > 0:
        prefix = torch.zeros((tgt_len, past_key_values_length), dtype=dtype, device=device)
        mask = torch.cat([prefix, mask], dim=-1)
    return mask[None, None, :, :]


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int) -> torch.Tensor:
    bsz, src_len = mask.shape
    expanded = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted = 1.0 - expanded
    return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)


class OneLayerPredictor(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.layer = copy.deepcopy(decoder.layers[-1])
        self.layer_norm = copy.deepcopy(decoder.layer_norm)
        if hasattr(self.layer, "layer_idx"):
            self.layer.layer_idx = 0
        if hasattr(self.layer, "self_attn"):
            self.layer.self_attn.layer_idx = 0
        if hasattr(self.layer, "encoder_attn"):
            self.layer.encoder_attn.layer_idx = 0

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        encoder_hidden_states: torch.Tensor,
        past_key_values: EncoderDecoderCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor:
        seq_len = hidden_states.size(1)
        combined_mask = None
        if seq_len > 1:
            combined_mask = _make_causal_mask(
                (hidden_states.size(0), seq_len),
                hidden_states.dtype,
                hidden_states.device,
            )
        if attention_mask is not None:
            expanded_mask = _expand_mask(
                attention_mask,
                hidden_states.dtype,
                tgt_len=seq_len,
            )
            combined_mask = (
                expanded_mask if combined_mask is None else expanded_mask + combined_mask
            )
        layer_outputs = self.layer(
            hidden_states,
            attention_mask=combined_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=False,
        )
        hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
        return self.layer_norm(hidden_states)


class MultiTokenTurboPredictor(nn.Module):
    requires_processor = True

    def __init__(
        self,
        checkpoint_path: str | Path,
        dtype: torch.dtype = torch.float32,
        strict: bool = True,
    ):
        super().__init__()
        checkpoint = self._load_checkpoint(Path(checkpoint_path))
        self.model_name = checkpoint.get("model_name", "openai/whisper-large-v3-turbo")
        self.future_tokens = int(checkpoint.get("future_tokens", 0))
        self.base = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
        )
        self.base.config.use_cache = True
        self.base.generation_config.do_sample = False
        self.base.generation_config.return_timestamps = False
        self.base.generation_config.use_cache = True
        self.mtp_heads = nn.ModuleList(
            [OneLayerPredictor(self.base.model.decoder) for _ in range(self.future_tokens)]
        )
        self.load_info = self.load_state_dict(checkpoint["state_dict"], strict=strict)
        self.last_generation_stats: dict[str, int] = {
            "successful_predictions": 0,
            "proposed_predictions": 0,
            "accepted_tokens": 0,
            "steps": 0,
        }
        self.to(dtype=dtype)

    @property
    def model(self):
        return self.base.model

    @property
    def config(self):
        return self.base.config

    def _load_checkpoint(self, checkpoint_path: Path) -> dict[str, object]:
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "mtp.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
            raise TypeError(f"Unsupported MTP checkpoint payload: {type(checkpoint)!r}")
        return checkpoint

    def _encode(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None):
        return self.base.model.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state

    def _prompt_ids(self, processor, language: str | None, task: str) -> list[int]:
        prompt = [self.base.config.decoder_start_token_id]
        if language is None:
            return prompt
        prompt.extend(
            token_id
            for _, token_id in sorted(
                processor.get_decoder_prompt_ids(language=language, task=task)
            )
        )
        return prompt

    def _resolve_max_length(
        self,
        processor,
        language: str | None,
        task: str,
        max_new_tokens: int | None,
        max_length: int | None,
    ) -> int:
        ceiling = (
            getattr(self.base.config, "max_target_positions", None)
            or self.base.generation_config.max_length
        )
        if max_length is not None:
            return min(max_length, ceiling)
        prompt_len = len(self._prompt_ids(processor, language, task))
        if max_new_tokens is not None:
            return min(prompt_len + max_new_tokens, ceiling)
        return ceiling

    def _new_cache(self) -> EncoderDecoderCache:
        return EncoderDecoderCache(
            DynamicCache(config=self.base.config),
            DynamicCache(config=self.base.config),
        )

    def _crop_self_cache(self, cache: EncoderDecoderCache, max_length: int) -> None:
        cache.self_attention_cache.crop(max_length)

    def _prime_heads(
        self,
        base_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> tuple[list[EncoderDecoderCache], list[torch.Tensor]]:
        caches: list[EncoderDecoderCache] = []
        outputs: list[torch.Tensor] = []
        for head in self.mtp_heads:
            cache = self._new_cache()
            last_hidden = base_hidden_states[:, -1, :]
            for token_idx in range(base_hidden_states.size(1)):
                last_hidden = head(
                    base_hidden_states[:, token_idx : token_idx + 1, :],
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=cache,
                    use_cache=True,
                )[:, -1, :]
            caches.append(cache)
            outputs.append(last_hidden)
        return caches, outputs

    def _advance_heads(
        self,
        head_caches: list[EncoderDecoderCache],
        base_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for head, cache in zip(self.mtp_heads, head_caches, strict=False):
            last_hidden = base_hidden_states[:, -1, :]
            for token_idx in range(base_hidden_states.size(1)):
                last_hidden = head(
                    base_hidden_states[:, token_idx : token_idx + 1, :],
                    attention_mask=None,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=cache,
                    use_cache=True,
                )[:, -1, :]
            outputs.append(last_hidden)
        return outputs

    def _apply_logit_rules(
        self,
        scores: torch.Tensor,
        *,
        suppress_tokens: list[int],
        begin_suppress_tokens: list[int],
        at_begin: bool,
    ) -> torch.Tensor:
        if suppress_tokens:
            scores[:, suppress_tokens] = torch.finfo(scores.dtype).min
        if at_begin and begin_suppress_tokens:
            scores[:, begin_suppress_tokens] = torch.finfo(scores.dtype).min
        return scores

    def _next_token(
        self,
        hidden_state: torch.Tensor,
        *,
        suppress_tokens: list[int],
        begin_suppress_tokens: list[int],
        at_begin: bool,
    ) -> torch.Tensor:
        scores = self.base.proj_out(hidden_state).clone()
        scores = self._apply_logit_rules(
            scores,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            at_begin=at_begin,
        )
        return scores.argmax(dim=-1, keepdim=True)

    @torch.no_grad()
    def generate(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        processor=None,
        language: str | None = None,
        task: str = "transcribe",
        max_new_tokens: int | None = None,
        max_length: int | None = None,
        **_: object,
    ) -> torch.Tensor:
        if processor is None:
            raise ValueError("MultiTokenTurboPredictor.generate requires a WhisperProcessor.")
        if input_features.size(0) != 1:
            raise ValueError("MultiTokenTurboPredictor.generate expects batch_size=1.")

        encoder_hidden_states = self._encode(input_features, attention_mask)
        ids = torch.tensor(
            [self._prompt_ids(processor, language, task)],
            device=input_features.device,
        )
        prompt_len = ids.size(1)
        max_length = self._resolve_max_length(
            processor,
            language,
            task,
            max_new_tokens,
            max_length,
        )
        eos_token_id = self.base.generation_config.eos_token_id
        suppress_tokens = list(getattr(self.base.generation_config, "suppress_tokens", None) or [])
        begin_suppress_tokens = list(
            getattr(self.base.generation_config, "begin_suppress_tokens", None) or []
        )
        base_cache = self._new_cache()
        base_outputs = self.base.model.decoder(
            input_ids=ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=base_cache,
            use_cache=True,
            return_dict=True,
        )
        current_base_hidden = base_outputs.last_hidden_state[:, -1, :]
        head_caches, current_head_outputs = self._prime_heads(
            base_outputs.last_hidden_state,
            encoder_hidden_states,
        )
        stats = {
            "successful_predictions": 0,
            "proposed_predictions": 0,
            "accepted_tokens": 0,
            "steps": 0,
            "full_generation_step_times": [],
        }

        while ids.size(1) < max_length:
            if input_features.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_start = time.perf_counter()

            block = [
                self._next_token(
                    current_base_hidden,
                    suppress_tokens=suppress_tokens,
                    begin_suppress_tokens=begin_suppress_tokens,
                    at_begin=ids.size(1) == prompt_len,
                )
            ]
            for predictor_hidden in current_head_outputs:
                block.append(
                    self._next_token(
                        predictor_hidden,
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=[],
                        at_begin=False,
                    )
                )

            remaining_tokens = max_length - ids.size(1)
            block = block[:remaining_tokens]
            for index, token in enumerate(block):
                if token.item() == eos_token_id:
                    block = block[: index + 1]
                    break

            accepted = [block[0]]
            speculative_tokens = block[1:]
            stats["proposed_predictions"] += len(speculative_tokens)
            stats["steps"] += 1

            if not speculative_tokens:
                commit_outputs = self.base.model.decoder(
                    input_ids=block[0],
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=base_cache,
                    use_cache=True,
                    return_dict=True,
                )
                new_base_hidden_states = commit_outputs.last_hidden_state
                current_base_hidden = new_base_hidden_states[:, -1, :]
            else:
                prefix_cache_len = base_cache.get_seq_length()
                verify_ids = torch.cat(block[:-1], dim=-1)
                verify_hidden = self.base.model.decoder(
                    input_ids=verify_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=base_cache,
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state
                fallback_token = None
                match_count = 0

                for index, speculative in enumerate(speculative_tokens):
                    verified = self._next_token(
                        verify_hidden[:, index, :],
                        suppress_tokens=suppress_tokens,
                        begin_suppress_tokens=[],
                        at_begin=False,
                    )
                    if not torch.equal(verified, speculative):
                        fallback_token = verified
                        break
                    accepted.append(speculative)
                    stats["successful_predictions"] += 1
                    match_count += 1

                if fallback_token is None:
                    final_token = block[-1]
                    commit_outputs = self.base.model.decoder(
                        input_ids=final_token,
                        encoder_hidden_states=encoder_hidden_states,
                        past_key_values=base_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    new_base_hidden_states = torch.cat(
                        [verify_hidden, commit_outputs.last_hidden_state],
                        dim=1,
                    )
                    current_base_hidden = commit_outputs.last_hidden_state[:, -1, :]
                    accepted.append(final_token)
                else:
                    accepted_prefix_len = 1 + match_count
                    self._crop_self_cache(base_cache, prefix_cache_len + accepted_prefix_len)
                    commit_outputs = self.base.model.decoder(
                        input_ids=fallback_token,
                        encoder_hidden_states=encoder_hidden_states,
                        past_key_values=base_cache,
                        use_cache=True,
                        return_dict=True,
                    )
                    new_base_hidden_states = torch.cat(
                        [
                            verify_hidden[:, :accepted_prefix_len, :],
                            commit_outputs.last_hidden_state,
                        ],
                        dim=1,
                    )
                    current_base_hidden = commit_outputs.last_hidden_state[:, -1, :]
                    accepted.append(fallback_token)

            accepted_ids = torch.cat(accepted, dim=-1)
            ids = torch.cat([ids, accepted_ids], dim=-1)
            stats["accepted_tokens"] += int(accepted_ids.size(1))

            if self.mtp_heads:
                current_head_outputs = self._advance_heads(
                    head_caches,
                    new_base_hidden_states,
                    encoder_hidden_states,
                )

            if accepted_ids[0, -1].item() == eos_token_id:
                if input_features.device.type == "cuda" and torch.cuda.is_available():
                    torch.cuda.synchronize()
                stats["full_generation_step_times"].append(time.perf_counter() - step_start)
                self.last_generation_stats = stats
                return ids

            if input_features.device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            stats["full_generation_step_times"].append(time.perf_counter() - step_start)

        self.last_generation_stats = stats
        return ids

    def get_timer(self):
        return None

    def __call__(self):
        return self
