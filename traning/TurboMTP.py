import copy
import io

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL = "openai/whisper-large-v3-turbo"
LANGUAGE, TASK = "english", "transcribe"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


def load_audio(audio):
    source = io.BytesIO(audio["bytes"]) if audio.get("bytes") is not None else audio["path"]
    wav, sr = sf.read(source, dtype="float32")
    wav = torch.as_tensor(wav).float()
    if wav.ndim == 2:
        wav = wav.mean(dim=1)
    if sr != 16_000:
        wav = torchaudio.functional.resample(wav, sr, 16_000)
    return wav.numpy()


class TurboMTP(nn.Module):
    def __init__(self, future_tokens=3):
        super().__init__()
        base = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=dtype).to(device).eval()
        self.encoder = base.model.encoder
        self.decoder = base.model.decoder
        self.proj_out = base.proj_out
        self.start_id = base.config.decoder_start_token_id
        self.eos_id = base.generation_config.eos_token_id
        self.mtp = nn.ModuleList([self._clone_one_layer_decoder(base.model.decoder) for _ in range(future_tokens)])

    def _clone_one_layer_decoder(self, decoder):
        head = copy.deepcopy(decoder)
        layer = copy.deepcopy(decoder.layers[-1])
        if hasattr(layer, "layer_idx"):
            layer.layer_idx = 0
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = 0
        if hasattr(layer, "encoder_attn"):
            layer.encoder_attn.layer_idx = 0
        head.layers = nn.ModuleList([layer])
        head.config.decoder_layers = 1
        return head

    def encode(self, input_features, attention_mask=None):
        return self.encoder(input_features=input_features, attention_mask=attention_mask, return_dict=True).last_hidden_state

    def make_targets(self, labels):
        targets = []
        for shift in range(len(self.mtp) + 1):
            target = labels.new_full(labels.shape, -100)
            if shift == 0:
                target.copy_(labels)
            elif labels.size(1) > shift:
                target[:, :-shift] = labels[:, shift:]
            targets.append(target)
        return targets

    def forward(self, input_features, decoder_input_ids, attention_mask=None, decoder_attention_mask=None):
        enc = self.encode(input_features, attention_mask)
        dec_mask = decoder_attention_mask if decoder_attention_mask is not None else torch.ones_like(decoder_input_ids)
        hidden = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=dec_mask,
            encoder_hidden_states=enc,
            return_dict=True,
        ).last_hidden_state
        logits = [self.proj_out(hidden)]
        state = hidden
        for head in self.mtp:
            state = head(
                inputs_embeds=state,
                attention_mask=dec_mask,
                encoder_hidden_states=enc,
                return_dict=True,
            ).last_hidden_state
            logits.append(self.proj_out(state))
        return logits

    def generate(self, input_features, prompt_ids, attention_mask=None, max_steps=32):
        with torch.no_grad():
            enc = self.encode(input_features, attention_mask)
            ids = torch.tensor([prompt_ids], device=device)
            for step in range(max_steps):
                dec_mask = torch.ones_like(ids)
                hidden = self.decoder(
                    input_ids=ids,
                    attention_mask=dec_mask,
                    encoder_hidden_states=enc,
                    return_dict=True,
                ).last_hidden_state
                block = [self.proj_out(hidden[:, -1, :]).argmax(dim=-1, keepdim=True)]
                state = hidden
                for head in self.mtp:
                    state = head(
                        inputs_embeds=state,
                        attention_mask=dec_mask,
                        encoder_hidden_states=enc,
                        return_dict=True,
                    ).last_hidden_state
                    block.append(self.proj_out(state[:, -1, :]).argmax(dim=-1, keepdim=True))

                accepted = [block[0]]
                verify_ids = torch.cat([ids, block[0]], dim=-1)
                for speculative in block[1:]:
                    verify_hidden = self.decoder(
                        input_ids=verify_ids,
                        attention_mask=torch.ones_like(verify_ids),
                        encoder_hidden_states=enc,
                        return_dict=True,
                    ).last_hidden_state
                    verified = self.proj_out(verify_hidden[:, -1, :]).argmax(dim=-1, keepdim=True)
                    if not torch.equal(verified, speculative):
                        break
                    accepted.append(speculative)
                    verify_ids = torch.cat([verify_ids, speculative], dim=-1)

                ids = torch.cat([ids] + accepted, dim=-1)
                proposed = [token.item() for token in block]
                accepted_ids = [token.item() for token in accepted]
                print(f"step={step:02d} proposed={proposed} accepted={accepted_ids}")
                if any(token == self.eos_id for token in accepted_ids):
                    break
        return ids


ds = load_dataset("distil-whisper/earnings22", "chunked", split="test", streaming=True)
row = next(iter(ds.cast_column("audio", Audio(decode=False))))
audio = load_audio(row["audio"])
text = " ".join(row["transcription"].strip().lower().split())

processor = WhisperProcessor.from_pretrained(MODEL)
processor.tokenizer.set_prefix_tokens(language=LANGUAGE, task=TASK)
features = processor.feature_extractor(
    [audio],
    sampling_rate=16_000,
    padding="max_length",
    max_length=processor.feature_extractor.n_samples,
    truncation=True,
    return_attention_mask=True,
    return_tensors="pt",
)
input_features = features.input_features.to(device)
attention_mask = features.attention_mask.to(device)

model = TurboMTP(future_tokens=3)
prompt_ids = [model.start_id] + [token_id for _, token_id in sorted(processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK))]
pred_ids = model.generate(input_features, prompt_ids, attention_mask)
pred = " ".join(processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip().lower().split())
print("ref:", text)
print("hyp:", pred)
