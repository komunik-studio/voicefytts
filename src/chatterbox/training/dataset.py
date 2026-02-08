
import os
import csv
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm

class VoicefyDataset(Dataset):
    """
    Dataset for fine-tuning VoicefyTTS T3 (LLaMA) model.
    Supports LJSpeech format.
    """
    def __init__(self, data_path, tokenizer, max_duration=15.0, min_duration=1.0, sample_rate=24000):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.sample_rate = sample_rate
        
        self.wavs_dir = self.data_path / "wavs"
        self.metadata_path = self.data_path / "metadata.csv"
        
        self.items = self._load_metadata()
        
    def _load_metadata(self):
        items = []
        if not self.metadata_path.exists():
            print(f"Metadata not found at {self.metadata_path}. Creating dataset from wavs ignoring text (not recommended for T3).")
            # Fallback logic if needed, but T3 needs text
            return items

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                # LJSpeech: id|text|normalized_text
                # Voicefy: id|text|speaker (optional)
                file_id = row[0]
                text = row[1]
                
                wav_path = self.wavs_dir / f"{file_id}.wav"
                if wav_path.exists():
                     items.append({
                        "id": file_id,
                        "wav_path": str(wav_path),
                        "text": text
                    })
        return items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Load Audio (for duration checking and potential S3Gen training later)
        # For T3 fine-tuning we mainly need text -> discrete codes mapping
        # But we need pre-extracted codes. 
        # For now, let's assume we implement on-the-fly extraction OR loading pre-computed codes.
        # This implementation assumes we will have to Tokenize Text.
        
        text = item["text"]
        text_tokens = self.tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        
        # Audio loading
        wav, sr = torchaudio.load(item["wav_path"])
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)(wav)
            
        # We need discrete speech tokens for T3 training. 
        # Ideally these are pre-computed. 
        # If not, we would need S3Gen encoder here.
        # For this skeleton, we will return wav and text, and assume the Collator or Train loop handles S3Gen encoding if not cached.
        
        return {
            "id": item["id"],
            "text": text,
            "text_tokens": text_tokens,
            "wav": wav,
            "wav_path": item["wav_path"]
        }
