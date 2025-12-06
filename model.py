"""
Improved Fusion Model with cross-attention and gating
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, AutoModel


class ImprovedFusionRegressor(nn.Module):
    """
    Dual-backbone model with:
    - CLIP for image and text encoding
    - DistilBERT for enhanced text understanding
    - Cross-attention between modalities
    - Gating mechanism for modality weighting
    - Auxiliary classification head
    """
    
    def __init__(
        self, 
        clip_id: str, 
        distil_id: str, 
        distil_proj_dim: int = 256, 
        head_hidden_mult: float = 2.0,
        num_price_bins: int = 10
    ):
        super().__init__()
        
        # Load CLIP model
        self.clip = CLIPModel.from_pretrained(clip_id)
        clip_dim = self.clip.config.projection_dim
        
        # Load DistilBERT model
        try:
            self.distil = AutoModel.from_pretrained(distil_id)
            self.distil_id_used = distil_id
        except Exception:
            self.distil = AutoModel.from_pretrained("distilbert-base-uncased")
            self.distil_id_used = "distilbert-base-uncased"
        
        distil_hidden = self.distil.config.dim
        
        # Projection layers to common dimension
        self.img_proj = nn.Linear(clip_dim, distil_proj_dim)
        self.txt_proj = nn.Linear(clip_dim, distil_proj_dim)
        self.distil_proj = nn.Sequential(
            nn.Linear(distil_hidden, distil_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(distil_hidden, distil_proj_dim),
        )
        
        # Cross-attention between modalities
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=distil_proj_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Gating mechanism for modality importance
        self.gate = nn.Sequential(
            nn.Linear(distil_proj_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )
        
        # Main regression head
        fused_dim = distil_proj_dim * 3
        hidden = int(fused_dim * head_hidden_mult)
        
        self.head_layer1 = nn.Linear(fused_dim, hidden)
        self.head_norm1 = nn.LayerNorm(hidden)
        self.head_layer2 = nn.Linear(hidden, hidden)
        self.head_norm2 = nn.LayerNorm(hidden)
        self.head_out = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(0.1)
        
        # Auxiliary classification head
        self.num_price_bins = num_price_bins
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, num_price_bins)
        )
        
    def forward(
        self,
        clip_input_ids,
        clip_attention_mask,
        pixel_values: Optional[torch.Tensor],
        img_missing: torch.Tensor,
        distil_ids1,
        distil_att1,
        distil_ids2,
        distil_att2,
    ):
        # Extract CLIP text features
        txt_feat = self.clip.get_text_features(
            input_ids=clip_input_ids,
            attention_mask=clip_attention_mask
        )
        
        # Extract CLIP image features
        if pixel_values is not None:
            vision_dtype = next(self.clip.vision_model.parameters()).dtype
            img_feat = self.clip.get_image_features(
                pixel_values=pixel_values.to(dtype=vision_dtype)
            )
            # Zero out features for missing images
            if img_missing.any():
                img_feat = img_feat * (1.0 - img_missing.unsqueeze(1).float())
        else:
            img_feat = torch.zeros_like(txt_feat)
        
        # Extract DistilBERT features (two views for contrastive learning)
        out1 = self.distil(input_ids=distil_ids1, attention_mask=distil_att1)
        out2 = self.distil(input_ids=distil_ids2, attention_mask=distil_att2)
        cls1 = out1.last_hidden_state[:, 0, :]
        cls2 = out2.last_hidden_state[:, 0, :]
        
        # Project to common dimension
        z_img = self.img_proj(img_feat)
        z_txt = self.txt_proj(txt_feat)
        z_d1 = self.distil_proj(cls1)
        z_d2 = self.distil_proj(cls2)
        
        # Cross-attention between modalities
        features = torch.stack([z_img, z_txt, z_d1], dim=1)  # (B, 3, proj_dim)
        attended, _ = self.cross_attn(features, features, features)
        
        # Extract and normalize attended features
        z_img_att = attended[:, 0, :]
        z_txt_att = attended[:, 1, :]
        z_d1_att = attended[:, 2, :]
        
        z_img_norm = F.normalize(z_img_att, dim=-1)
        z_txt_norm = F.normalize(z_txt_att, dim=-1)
        z_d1_norm = F.normalize(z_d1_att, dim=-1)
        z_d2_norm = F.normalize(z_d2, dim=-1)
        
        # Compute gating weights
        concat_for_gate = torch.cat([z_img_norm, z_txt_norm, z_d1_norm], dim=-1)
        gate_weights = self.gate(concat_for_gate)  # (B, 3)
        
        # Apply gating
        z_img_gated = z_img_norm * gate_weights[:, 0:1]
        z_txt_gated = z_txt_norm * gate_weights[:, 1:2]
        z_d1_gated = z_d1_norm * gate_weights[:, 2:3]
        
        # Fuse features
        fused = torch.cat([z_img_gated, z_txt_gated, z_d1_gated], dim=-1)
        
        # Regression head with residual connections
        h = F.gelu(self.head_norm1(self.head_layer1(fused)))
        h = self.dropout(h)
        h_res = self.head_layer2(h)
        h = F.gelu(self.head_norm2(h_res + h))
        h = self.dropout(h)
        pred_log2 = self.head_out(h).squeeze(-1)
        
        # Auxiliary classification
        class_logits = self.classifier(fused)
        
        return pred_log2, class_logits, z_img_norm, z_txt_norm, z_d1_norm, z_d2_norm, gate_weights