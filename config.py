"""
Configuration module for MCF
Handles all hyperparameters and environment variables
"""
import os
from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    clip_id: str = "openai/clip-vit-large-patch14"
    distil_id: str = "distilbert-base-uncased-finetuned-sst-2-english"
    distil_proj_dim: int = 256
    head_hidden_mult: float = 2.0
    num_price_bins: int = 10
    freeze_clip: bool = False
    freeze_distil: bool = False


@dataclass
class DataConfig:
    """Data processing configuration"""
    dataset_folder: str = '/kaggle/input/amazon-ml-challenge-2025/student_resource/dataset'
    text_col: str = 'catalog_content'
    img_col: str = 'image_link'
    price_col: str = 'price'
    val_frac: float = 0.01
    max_len_clip: int = 64
    max_len_distil: int = 192
    img_missing_policy: Literal['zero', 'text_only', 'drop'] = 'zero'
    min_price: float = 1e-6
    seed: int = 42


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    batch_size: int = 16
    epochs: int = 3
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    grad_accum: int = 1
    max_grad_norm: float = 1.0
    fp16: bool = True
    early_stop_rounds: int = 5
    
    # Loss weights
    alpha_clip_nce: float = 0.20
    alpha_txt_nce: float = 0.10
    alpha_cls: float = 0.30
    alpha_gate_reg: float = 0.10
    
    # Loss parameters
    loss_type: Literal['focal_huber', 'log_cosh', 'huber'] = 'focal_huber'
    huber_delta: float = 1.0
    tau: float = 0.07
    focal_gamma: float = 2.0
    word_mask_p: float = 0.06


@dataclass
class Config:
    """Master configuration class"""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    output_dir: str = "price_clip+distil_fusionv2_improved"
    
    @classmethod
    def from_env(cls):
        """Create configuration from environment variables"""
        model_config = ModelConfig(
            clip_id=os.environ.get("CLIP_ID", "openai/clip-vit-large-patch14"),
            distil_id=os.environ.get("DISTIL_ID", "distilbert-base-uncased-finetuned-sst-2-english"),
            num_price_bins=int(os.environ.get("NUM_PRICE_BINS", "10")),
            freeze_clip=os.environ.get("FREEZE_CLIP", "false").lower() == "true",
            freeze_distil=os.environ.get("FREEZE_DISTIL", "false").lower() == "true",
        )
        
        data_config = DataConfig(
            dataset_folder=os.environ.get("DATASET_FOLDER", "/kaggle/input/amazon-ml-challenge-2025/student_resource/dataset"),
            val_frac=float(os.environ.get("VAL_FRAC", "0.01")),
            max_len_clip=int(os.environ.get("MAX_LEN_CLIP", "64")),
            max_len_distil=int(os.environ.get("MAX_LEN_DISTIL", "192")),
            img_missing_policy=os.environ.get("IMG_MISSING_POLICY", "zero").lower(),
            min_price=float(os.environ.get("MIN_PRICE", "1e-6")),
            seed=int(os.environ.get("SEED", "42")),
        )
        
        training_config = TrainingConfig(
            batch_size=int(os.environ.get("BATCH_SIZE", "16")),
            epochs=int(os.environ.get("EPOCHS", "3")),
            lr=float(os.environ.get("LR", "2e-5")),
            weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.01")),
            warmup_ratio=float(os.environ.get("WARMUP_RATIO", "0.06")),
            grad_accum=int(os.environ.get("GRAD_ACCUM", "1")),
            max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "1.0")),
            fp16=os.environ.get("FP16", "true").lower() == "true",
            early_stop_rounds=int(os.environ.get("EARLY_STOP_ROUNDS", "5")),
            alpha_clip_nce=float(os.environ.get("ALPHA_CLIP_NCE", "0.20")),
            alpha_txt_nce=float(os.environ.get("ALPHA_TXT_NCE", "0.10")),
            alpha_cls=float(os.environ.get("ALPHA_CLS", "0.30")),
            alpha_gate_reg=float(os.environ.get("ALPHA_GATE_REG", "0.10")),
            loss_type=os.environ.get("LOSS_TYPE", "focal_huber"),
            huber_delta=float(os.environ.get("HUBER_DELTA", "1.0")),
            tau=float(os.environ.get("TAU", "0.07")),
            focal_gamma=float(os.environ.get("FOCAL_GAMMA", "2.0")),
            word_mask_p=float(os.environ.get("WORD_MASK_P", "0.06")),
        )
        
        output_dir = os.environ.get("OUTPUT_DIR", "price_clip+distil_fusionv2_improved")
        os.makedirs(output_dir, exist_ok=True)
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config,
            output_dir=output_dir
        )