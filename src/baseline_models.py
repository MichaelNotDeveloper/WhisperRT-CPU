from transformers import WhisperForConditionalGeneration
import torch

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
        compile_mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = False,
    ):
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )

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