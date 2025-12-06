"""
Main training script for MCF
"""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer

from config import Config
from utils import set_seed, log2_price, split_train_val, count_parameters, prepare_data
from dataset import DualBackboneDataset, DualCollate
from model import ImprovedFusionRegressor
from trainer import Trainer
from data_download import download_images


def main():
    # Load configuration
    config = Config.from_env()
    
    # Set random seed
    set_seed(config.data.seed)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(os.path.join(config.data.dataset_folder, 'train.csv'))
    test_df = pd.read_csv(os.path.join(config.data.dataset_folder, 'test.csv'))
    
    
    # Prepare data
    print("\n Cleaning data...")
    df = prepare_data(train_df, config.data.price_col, config.data.min_price)
    
    # Split train/val
    df_tr, df_va = split_train_val(df, frac_val=config.data.val_frac, seed=config.data.seed)
    print(f"Split: train={len(df_tr)} | valid={len(df_va)}")
    
    # Transform targets
    y_tr_log = log2_price(df_tr[config.data.price_col].values, config.data.min_price)
    y_va_log = log2_price(df_va[config.data.price_col].values, config.data.min_price)
    
    # Load tokenizers and processors
    print("\nLoading tokenizers...")
    clip_proc = AutoProcessor.from_pretrained(config.model.clip_id)
    
    try:
        distil_tok = AutoTokenizer.from_pretrained(config.model.distil_id, use_fast=True)
        used_distil_tok_id = config.model.distil_id
    except Exception:
        print("Could not load specified DistilBERT tokenizer, using default")
        distil_tok = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=True)
        used_distil_tok_id = "distilbert-base-uncased"
    
    # Add mask token if needed
    if distil_tok.mask_token is None:
        distil_tok.add_special_tokens({"mask_token": "[MASK]"})
    
    # Create datasets
    print("\n Creating datasets...")
    train_ds = DualBackboneDataset(
        df=df_tr,
        text_col=config.data.text_col,
        img_col=config.data.img_col,
        y_log2=y_tr_log,
        clip_processor=clip_proc,
        distil_tok=distil_tok,
        max_len_clip=config.data.max_len_clip,
        max_len_distil=config.data.max_len_distil,
        policy=config.data.img_missing_policy,
        word_mask_p=config.training.word_mask_p,
        training=True
    )
    
    val_ds = DualBackboneDataset(
        df=df_va,
        text_col=config.data.text_col,
        img_col=config.data.img_col,
        y_log2=y_va_log,
        clip_processor=clip_proc,
        distil_tok=distil_tok,
        max_len_clip=config.data.max_len_clip,
        max_len_distil=config.data.max_len_distil,
        policy=config.data.img_missing_policy,
        word_mask_p=0.0,  # No augmentation for validation
        training=False
    )
    
    # Create collate function
    collate = DualCollate(
        clip_tokenizer=clip_proc.tokenizer,
        distil_pad_id=distil_tok.pad_token_id if distil_tok.pad_token_id is not None else 0
    )
    
    # Create dataloaders
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate
    )
    
    # Warmup to count missing images
    if len(train_loader) > 0:
        _ = next(iter(train_loader))
    if len(val_loader) > 0:
        _ = next(iter(val_loader))
    
    print(f"Missing images (train/val): {train_ds.missing_img_count}/{val_ds.missing_img_count}")
    print(f"Dropped images (train/val): {train_ds.dropped_missing}/{val_ds.dropped_missing}")
    
    # Create model
    print("\nCreating model...")
    model = ImprovedFusionRegressor(
        config.model.clip_id,
        used_distil_tok_id,
        distil_proj_dim=config.model.distil_proj_dim,
        head_hidden_mult=config.model.head_hidden_mult,
        num_price_bins=config.model.num_price_bins
    ).to(device)
    
    # Freeze layers if specified
    if config.model.freeze_clip:
        print(" Freezing CLIP backbone")
        for p in model.clip.parameters():
            p.requires_grad = False
    
    if config.model.freeze_distil:
        print("Freezing DistilBERT backbone")
        for p in model.distil.parameters():
            p.requires_grad = False
    
    print(f"Device: {device}")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Loss type: {config.training.loss_type}")
    
    # Create trainer
    print("\n Starting training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    best_smape = trainer.train(y_va_log)
    
    print(f"\n{'='*60}")
    print(f"Training complete! Best SMAPE: {best_smape:.3f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()