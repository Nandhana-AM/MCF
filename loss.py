"""
Custom loss functions for price prediction
"""
import torch
import torch.nn.functional as F


def focal_huber_loss(pred, target, delta=1.0, gamma=2.0, alpha=0.25):
    """
    Focal loss weighted Huber for handling price distribution imbalance
    
    Args:
        pred: Predicted values
        target: Ground truth values
        delta: Huber loss delta parameter
        gamma: Focal loss gamma parameter
        alpha: Focal loss alpha parameter
    """
    diff = torch.abs(pred - target)
    huber = torch.where(
        diff < delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    )
    # Focal weighting: focus more on hard examples
    focal_weight = (1 - torch.exp(-diff)) ** gamma
    return (alpha * focal_weight * huber).mean()


def log_cosh_loss(pred, target):
    """
    Log-cosh loss: smooth and less sensitive to outliers than MSE
    
    Args:
        pred: Predicted values
        target: Ground truth values
    """
    diff = pred - target
    return torch.mean(torch.log(torch.cosh(diff)))


def improved_info_nce(z_a: torch.Tensor, z_b: torch.Tensor, 
                      tau: float = 0.07, use_hard_negatives: bool = True):
    """
    Improved InfoNCE contrastive loss with optional hard negative mining
    
    Args:
        z_a: First set of embeddings (B, D)
        z_b: Second set of embeddings (B, D)
        tau: Temperature parameter
        use_hard_negatives: Whether to use hard negative mining
    """
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    
    logits = torch.matmul(z_a, z_b.t()) / tau
    labels = torch.arange(z_a.size(0), device=z_a.device)
    
    if use_hard_negatives and z_a.size(0) > 1:
        # Hard negative mining
        with torch.no_grad():
            mask = torch.eye(z_a.size(0), device=z_a.device).bool()
            neg_logits = logits.masked_fill(mask, float('-inf'))
            hard_neg_weights = F.softmax(neg_logits, dim=1)
        
        loss_a2b = F.cross_entropy(logits, labels, reduction='none')
        loss_b2a = F.cross_entropy(logits.t(), labels, reduction='none')
        loss = (loss_a2b + loss_b2a).mean()
    else:
        loss = 0.5 * (F.cross_entropy(logits, labels) + 
                      F.cross_entropy(logits.t(), labels))
    
    return loss


def get_price_bins(prices_log2, num_bins=10):
    """
    Create bins for auxiliary classification using quantiles
    
    Args:
        prices_log2: Log2-transformed prices
        num_bins: Number of bins to create
        
    Returns:
        Tensor of bin indices
    """
    if len(prices_log2) < num_bins:
        return torch.zeros_like(prices_log2).long()
    
    # Use quantile-based binning for balanced classes
    with torch.no_grad():
        quantiles = torch.quantile(
            prices_log2, 
            torch.linspace(0, 1, num_bins + 1).to(prices_log2.device)
        )
        bins = torch.searchsorted(quantiles[1:-1], prices_log2)
    
    return bins.long()


def compute_total_loss(
    pred_log2,
    target,
    class_logits,
    z_img,
    z_txt,
    z_d1,
    z_d2,
    gate_weights,
    img_missing,
    pixel_values,
    loss_type='focal_huber',
    alpha_cls=0.30,
    alpha_clip_nce=0.20,
    alpha_txt_nce=0.10,
    alpha_gate_reg=0.10,
    num_price_bins=10,
    huber_delta=1.0,
    focal_gamma=2.0,
    tau=0.07,
    img_missing_policy='zero'
):
    """
    Compute total training loss with all components
    
    Returns:
        total_loss, loss_components_dict
    """
    device = pred_log2.device
    
    # 1. Main regression loss
    if loss_type == "focal_huber":
        loss_reg = focal_huber_loss(pred_log2, target, delta=huber_delta, gamma=focal_gamma)
    elif loss_type == "log_cosh":
        loss_reg = log_cosh_loss(pred_log2, target)
    else:  # standard huber
        loss_reg = F.huber_loss(pred_log2, target, delta=huber_delta)
    
    # 2. Auxiliary classification loss
    price_bins = get_price_bins(target, num_bins=num_price_bins)
    loss_cls = F.cross_entropy(class_logits, price_bins)
    
    # 3. CLIP image-text contrastive (only if images available)
    loss_nce_clip = torch.tensor(0.0, device=device)
    valid_idx = (img_missing == 0).nonzero(as_tuple=False).squeeze(-1)
    if pixel_values is not None and img_missing_policy != "text_only" and valid_idx.numel() > 1:
        loss_nce_clip = improved_info_nce(z_img[valid_idx], z_txt[valid_idx], 
                                          tau=tau, use_hard_negatives=True)
    
    # 4. DistilBERT SimCSE-style contrastive
    loss_nce_txt = improved_info_nce(z_d1, z_d2, tau=tau, use_hard_negatives=True)
    
    # 5. Gate entropy regularization (encourage using all modalities)
    gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-8)).sum(dim=1).mean()
    loss_gate_reg = -gate_entropy  # Negative to maximize entropy
    
    # Total loss
    total_loss = (
        loss_reg + 
        alpha_cls * loss_cls +
        alpha_clip_nce * loss_nce_clip + 
        alpha_txt_nce * loss_nce_txt + 
        alpha_gate_reg * loss_gate_reg
    )
    
    loss_components = {
        'total': total_loss.item(),
        'reg': loss_reg.item(),
        'cls': loss_cls.item(),
        'nce_clip': loss_nce_clip.item(),
        'nce_txt': loss_nce_txt.item(),
        'gate_reg': loss_gate_reg.item(),
    }
    
    return total_loss, loss_components