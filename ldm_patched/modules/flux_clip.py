import torch
from transformers import CLIPTokenizer, T5TokenizerFast, CLIPTextModel, T5EncoderModel
import ldm_patched.modules.ops

class FluxTokenizer:
    def __init__(self, embedding_directory=None):
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer")
        self.t5_tokenizer = T5TokenizerFast.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2")

    def tokenize_with_weights(self, text, return_word_ids=False):
        # This is a simplified implementation. Proper weight handling is complex.
        clip_tokens = self.clip_tokenizer(text, max_length=self.clip_tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5_tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "clip_input_ids": clip_tokens.input_ids.squeeze(0),
            "t5_input_ids": t5_tokens.input_ids.squeeze(0)
        }

class FluxClipModel(torch.nn.Module):
    def __init__(self, device="cpu", dtype=torch.float32, **kwargs):
        super().__init__()
        self.clip_l = CLIPTextModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder", torch_dtype=dtype).to(device)
        self.clip_t5 = T5EncoderModel.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="text_encoder_2", torch_dtype=dtype).to(device)

    def reset_clip_layer(self):
        pass

    def encode_token_weights(self, tokens):
        clip_output = self.clip_l(input_ids=tokens["clip_input_ids"].unsqueeze(0).to(self.clip_l.device))
        clip_embeds = clip_output.last_hidden_state
        pooled_output = clip_output.pooler_output

        t5_output = self.clip_t5(input_ids=tokens["t5_input_ids"].unsqueeze(0).to(self.clip_t5.device))
        t5_embeds = t5_output.last_hidden_state

        # Concatenate along the sequence dimension
        cond = torch.cat([clip_embeds, t5_embeds], dim=-1)
        
        return cond, pooled_output

    def forward(self, tokens):
        return self.encode_token_weights(tokens)

    def load_sd(self, sd):
        # This needs to be implemented to load weights from a checkpoint
        return super().load_state_dict(sd)