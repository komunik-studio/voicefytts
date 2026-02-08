
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from safetensors.torch import load_file, save_file
import logging

from ..models.t3.t3 import T3
from ..models.s3gen.s3gen import S3Token2Wav
from ..models.voice_encoder.voice_encoder import VoiceEncoder
from ..models.tokenizers import EnTokenizer as Tokenizer
from .config import TrainingConfig
from .dataset import VoicefyDataset
from .collate import VoicefyCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train(config: TrainingConfig):
    # Setup
    device = torch.device(config.device)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Models
    logger.info(f"Loading models from {config.model_name}...")
    # NOTE: In a real scenario, we would load from a specific checkpoint path.
    # Here we assume model_name is a path or we load standard checkpoints.
    # For this toolkit, we assume the user provides the base checkpoint path in config.model_name
    ckpt_dir = Path(config.model_name)
    
    # Load Tokenizer
    tokenizer = Tokenizer(lang="english") # Default to english for now

    # Load T3 (Trainable)
    t3 = T3()
    if (ckpt_dir / "t3_cfg.safetensors").exists():
        t3.load_state_dict(load_file(ckpt_dir / "t3_cfg.safetensors"))
    t3.to(device)
    t3.train()
    
    # Load S3Gen (Frozen, for tokenizer)
    logger.info("Loading S3Gen for tokenization...")
    # We need validation reference dict for S3Gen init, but for tokenizer only it might be skippable
    # Or we load a dummy one. s3gen init requires some args.
    # Let's try loading it fully if possible, or just the tokenizer part if separated.
    # S3Token2Wav initializes S3Tokenizer internally.
    s3gen = S3Token2Wav(vocoder_type='hifigan') # Vocoder doesn't matter for tokenization
    if (ckpt_dir / "s3gen.safetensors").exists():
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
    s3gen.to(device)
    s3gen.eval()
    for p in s3gen.parameters():
        p.requires_grad = False
        
    # Load Voice Encoder (Frozen)
    logger.info("Loading Voice Encoder...")
    ve = VoiceEncoder()
    if (ckpt_dir / "ve.safetensors").exists():
        ve.load_state_dict(load_file(ckpt_dir / "ve.safetensors"))
    ve.to(device)
    ve.eval()
    for p in ve.parameters():
        p.requires_grad = False

    # 2. Dataset & Dataloader
    logger.info("Preparing dataset...")
    dataset = VoicefyDataset(
        data_path=config.train_data_path,
        tokenizer=tokenizer,
        sample_rate=24000 # S3Gen expects 24k input for mel, but Tokenizer expects 16k?
        # S3Token2Wav.inference/forward usually handles resampling if needed.
        # But wait, s3gen.tokenizer expects 16k?
        # In s3gen.py: ref_speech_tokens, ... = self.tokenizer(ref_wav_16.float())
        # So we should probably provide 16k audio to tokenizer?
        # Let's handle resampling in the loop.
    )
    
    collator = VoicefyCollator(tokenizer)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=config.num_workers,
        collate_fn=collator
    )

    # 3. Optimizer
    optimizer = optim.AdamW(t3.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(config.mixed_precision == "fp16"))

    # 4. Training Loop
    logger.info("Starting training...")
    global_step = 0
    t3.compiled = False # Ensure not using compiled model for training if not supported

    for epoch in range(config.num_epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            if batch is None: continue
            
            optimizer.zero_grad()
            
            # Move batch to device
            text_tokens = batch["text_tokens"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            wavs = batch["wavs"].to(device) # [B, T] @ 24k (from dataset default)
            wav_lengths = batch["wav_lengths"].to(device)
            
            # Prepare Inputs
            with torch.no_grad():
                # 1. Compute Speaker Embeddings (using VoiceEncoder)
                # VE expects 16k.
                wavs_16k = torch.nn.functional.interpolate(wavs.unsqueeze(1), scale_factor=16000/24000, mode='linear', align_corners=False).squeeze(1)
                
                # VoiceEncoder expects list of numpy arrays
                wavs_16k_np = [w.cpu().numpy() for w in wavs_16k]
                
                # Compute embeddings (on device inferred from ve)
                # as_spk=False returns (B, 256) utterance embeddings
                speaker_emb = ve.embeds_from_wavs(wavs_16k_np, sample_rate=16000, as_spk=False)
                speaker_emb = torch.from_numpy(speaker_emb).to(device)

                # Check T3Cond expectation. T3Cond expects [B, 1, dim]?
                # t3.cond_enc expects: speaker_emb (B, 1, 512/dim)
                if speaker_emb.dim() == 2:
                    speaker_emb = speaker_emb.unsqueeze(1)

                # 2. Compute Speech Tokens (Ground Truth)
                # s3gen.tokenizer expects 16k wav
                # output: tokens [B, T_tok], lens [B]
                speech_tokens, speech_token_lens = s3gen.tokenizer(wavs_16k)
                speech_tokens = speech_tokens.to(device)
                speech_token_lens = speech_token_lens.to(device)

                # Prepare T3 Condition
                # T3Cond usually needs cond_prompt_speech_tokens for prompting. 
                # For training, we are doing Teacher Forcing on the whole sequence?
                # Or is T3 trained to predict speech tokens from text + speaker emb?
                # T3 loss signature: t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens
                
                # We need to construct T3Cond.
                # For basic fine-tuning we might just condition on speaker embedding.
                # If we want to condition on a prompt, we'd slice the speech tokens? 
                # Standard T3 training usually trains on Text -> Speech Tokens relative to Speaker.
                
                from ..models.t3.modules.cond_enc import T3Cond
                t3_cond = T3Cond(
                    speaker_emb=speaker_emb,
                    # We can leave prompt tokens empty or use a slice if we want to simulate prompting
                    cond_prompt_speech_tokens=None 
                )
            
            # Mixing Precision
            with torch.cuda.amp.autocast(enabled=(config.mixed_precision != "no")):
                loss_text, loss_speech = t3.loss(
                    t3_cond=t3_cond,
                    text_tokens=text_tokens,
                    text_token_lens=text_lengths,
                    speech_tokens=speech_tokens,
                    speech_token_lens=speech_token_lens
                )
                loss = loss_speech + loss_text # Combined loss? Or just speech?
                # Usually we care about speech generation. But T3 trains both heads.
            
            # Backward
            scaler.scale(loss).backward()
            
            # Clip Grad Norm
            if config.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(t3.parameters(), config.max_grad_norm)
                
            scaler.step(optimizer)
            scaler.update()
            
            global_step += 1
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Save Checkpoint
            if global_step % config.save_steps == 0:
                save_path = output_dir / f"checkpoint-{global_step}"
                save_path.mkdir(exist_ok=True)
                save_file(t3.state_dict(), save_path / "t3_finetuned.safetensors")
                logger.info(f"Saved checkpoint to {save_path}")

    # Final Save
    save_file(t3.state_dict(), output_dir / "t3_finetuned_final.safetensors")
    logger.info("Training complete.")

if __name__ == "__main__":
    from .config import TrainingConfig
    config = TrainingConfig() 
    # User can override config values here or load from file
    train(config)
