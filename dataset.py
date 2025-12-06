"""
Dataset and collate function for dual-backbone model
"""
import os
import random
from typing import Optional, List
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


class DualBackboneDataset(Dataset):
    """Dataset for CLIP + DistilBERT dual backbone model"""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        text_col: str, 
        img_col: str,
        y_log2: Optional[np.ndarray],
        clip_processor: AutoProcessor,
        distil_tok: AutoTokenizer,
        max_len_clip: int,
        max_len_distil: int,
        policy: str,
        word_mask_p: float = 0.06,
        training: bool = True
    ):
        self.df = df.reset_index(drop=True).copy()
        self.text_col = text_col
        self.img_col = img_col
        self.y_log2 = y_log2
        self.proc = clip_processor
        self.tok = distil_tok
        self.max_len_clip = max_len_clip
        self.max_len_distil = max_len_distil
        self.policy = policy
        self.word_mask_p = word_mask_p
        self.training = training

        self.df[text_col] = self.df[text_col].fillna("").astype(str)

        # Handle missing images
        if policy == "drop":
            before = len(self.df)
            self.df = self.df[
                self.df[img_col].apply(
                    lambda p: isinstance(p, str) and len(p) > 0 and os.path.exists(p)
                )
            ]
            self.dropped_missing = before - len(self.df)
        else:
            self.dropped_missing = 0

        # Create dummy image for missing cases
        dummy = self.proc(images=Image.new("RGB", (224, 224)), return_tensors="pt")
        self._dummy_pixel = dummy["pixel_values"].squeeze(0)
        self.missing_img_count = 0

    def __len__(self):
        return len(self.df)

    def _load_image(self, pth: str):
        """Load image from path"""
        if isinstance(pth, str) and pth and os.path.exists(pth):
            try:
                return Image.open(pth).convert("RGB")
            except Exception:
                pass
        self.missing_img_count += 1
        return None

    def _tok_clip_text(self, text: str):
        """Tokenize text for CLIP"""
        enc = self.proc(
            text=[text], 
            truncation=True, 
            padding=False, 
            max_length=self.max_len_clip, 
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

    def _tok_distil(self, text: str):
        """Tokenize text for DistilBERT with augmentation"""
        base = self.tok(
            text, 
            truncation=True, 
            padding=False, 
            max_length=self.max_len_distil, 
            return_tensors="pt"
        )
        ids1 = base["input_ids"].clone()
        att1 = base["attention_mask"].clone()

        # Create augmented view for SimCSE-style contrastive learning
        ids2 = ids1.clone()
        att2 = att1.clone()

        if self.word_mask_p > 0 and self.tok.mask_token_id is not None:
            special = set(self.tok.all_special_ids)
            for i in range(ids2.size(1)):
                if ids2[0, i].item() in special:
                    continue
                if random.random() < self.word_mask_p:
                    ids2[0, i] = self.tok.mask_token_id

        return {
            "ids1": ids1.squeeze(0), "att1": att1.squeeze(0),
            "ids2": ids2.squeeze(0), "att2": att2.squeeze(0),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row[self.text_col]
        imgp = str(row[self.img_col]) if row[self.img_col] is not None else ""

        # Process text for CLIP
        clip_txt = self._tok_clip_text(text)

        # Process image
        img_missing = 0
        im = self._load_image(imgp)
        pixel_values = None
        
        if im is None:
            img_missing = 1
            if self.policy == "zero":
                pixel_values = self._dummy_pixel.clone()
        else:
            enc_img = self.proc(images=im, return_tensors="pt")
            pixel_values = enc_img["pixel_values"].squeeze(0)

        # Process text for DistilBERT
        distil = self._tok_distil(text)

        item = {
            "clip_input_ids": clip_txt["input_ids"],
            "clip_attention_mask": clip_txt["attention_mask"],
            "distil_ids1": distil["ids1"],
            "distil_att1": distil["att1"],
            "distil_ids2": distil["ids2"],
            "distil_att2": distil["att2"],
            "img_missing": torch.tensor(img_missing, dtype=torch.uint8),
        }
        
        if pixel_values is not None:
            item["pixel_values"] = pixel_values

        if self.y_log2 is not None:
            item["target"] = torch.tensor(self.y_log2[idx], dtype=torch.float32)
            
        return item


@dataclass
class DualCollate:
    """Collate function for batching"""
    clip_tokenizer: any
    distil_pad_id: int
    
    def __call__(self, batch):
        # Pad CLIP text
        clip_ids = [b["clip_input_ids"] for b in batch]
        clip_attn = [b["clip_attention_mask"] for b in batch]
        clip_padded = self.clip_tokenizer.pad(
            {"input_ids": clip_ids, "attention_mask": clip_attn},
            padding=True, 
            return_tensors="pt"
        )

        def pad_stack(tensors: List[torch.Tensor], pad_val: int):
            maxlen = max(t.size(0) for t in tensors)
            out = []
            for t in tensors:
                if t.size(0) < maxlen:
                    pad = torch.full((maxlen - t.size(0),), pad_val, dtype=t.dtype)
                    t = torch.cat([t, pad], dim=0)
                out.append(t.unsqueeze(0))
            return torch.cat(out, dim=0)

        # Pad DistilBERT tokens
        ids1 = pad_stack([b["distil_ids1"] for b in batch], self.distil_pad_id)
        att1 = pad_stack([b["distil_att1"] for b in batch], 0)
        ids2 = pad_stack([b["distil_ids2"] for b in batch], self.distil_pad_id)
        att2 = pad_stack([b["distil_att2"] for b in batch], 0)

        # Stack pixel values
        pixel_values = None
        if any("pixel_values" in b for b in batch):
            C, H, W = next(b["pixel_values"].shape for b in batch if "pixel_values" in b)
            pv = []
            for b in batch:
                pv.append(
                    b["pixel_values"] if "pixel_values" in b 
                    else torch.zeros((C, H, W), dtype=torch.float32)
                )
            pixel_values = torch.stack(pv, dim=0)

        res = {
            "clip_input_ids": clip_padded["input_ids"],
            "clip_attention_mask": clip_padded["attention_mask"],
            "distil_ids1": ids1,
            "distil_att1": att1,
            "distil_ids2": ids2,
            "distil_att2": att2,
            "img_missing": torch.stack([b["img_missing"] for b in batch], dim=0),
        }
        
        if pixel_values is not None:
            res["pixel_values"] = pixel_values
            
        if "target" in batch[0]:
            res["target"] = torch.stack([b["target"] for b in batch], dim=0)
            
        return res