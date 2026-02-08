
import torch
from torch.nn.utils.rnn import pad_sequence

class VoicefyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
        
    def __call__(self, batch):
        # Filter out None items if any
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
            
        ids = [b["id"] for b in batch]
        texts = [b["text"] for b in batch]
        
        # Pad text tokens
        text_tokens = [b["text_tokens"] for b in batch]
        text_tokens_padded = pad_sequence(text_tokens, batch_first=True, padding_value=self.pad_token_id)
        
        text_lengths = torch.tensor([len(t) for t in text_tokens])
        
        # For wavs, we usually pad to max length in batch
        wavs = [b["wav"].squeeze(0) for b in batch]
        wav_lengths = torch.tensor([w.shape[0] for w in wavs])
        wavs_padded = pad_sequence(wavs, batch_first=True, padding_value=0.0)

        return {
            "ids": ids,
            "texts": texts,
            "text_tokens": text_tokens_padded,
            "text_lengths": text_lengths,
            "wavs": wavs_padded,
            "wav_lengths": wav_lengths
        }
