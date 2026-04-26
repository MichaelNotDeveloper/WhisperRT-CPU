import copy
from pathlib import Path

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration


class BaselineSmall:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(dtype=torch.float32)
    def __call__(self):
        return self.model
    def get_timer(self):
        return None

class BaselineTurbo:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to(dtype=torch.float32)
    def __call__(self):
        return self.model
    def get_timer(self):
        return None

class BaselineLarge:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3").to(dtype=torch.float32)
    def __call__(self):
        return self.model
    def get_timer(self):
        return None


class BaselineTurboParams:
    def __init__(self):
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo").to(dtype=torch.float32)
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

class PrunedTurboDecoder():
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
            path for path in checkpoint_dir.iterdir() if path.is_file() and path.suffix in {".safetensors", ".bin"}
        )
        if len(matches) == 1:
            return matches[0]
        raise FileNotFoundError(
            f"Could not find weights inside {checkpoint_dir}. "
            "Expected a .safetensors or .bin file."
        )

    def _infer_decoder_layers(self, state_dict: dict[str, torch.Tensor]) -> int:
        prefix = "model.decoder.layers."
        indices = {
            int(key[len(prefix):].split(".", 1)[0])
            for key in state_dict
            if key.startswith(prefix)
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
            setattr(layer, "layer_idx", new_idx)
            if hasattr(layer, "self_attn"):
                setattr(layer.self_attn, "layer_idx", new_idx)
            if hasattr(layer, "encoder_attn"):
                setattr(layer.encoder_attn, "layer_idx", new_idx)

    def __call__(self):
        return self.model

    def get_timer(self):
        return None
