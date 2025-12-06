# MCF - Dual Backbone Model

A sophisticated deep learning model for predicting product prices using both text descriptions and images, combining CLIP and DistilBERT in a dual-backbone architecture with cross-attention and gating mechanisms.

## Features

- **Dual-Backbone Architecture**: Combines CLIP (for vision and text) with DistilBERT (for enhanced text understanding)
- **Cross-Attention Mechanism**: Allows modalities to attend to each other for better fusion
- **Adaptive Gating**: Learns to weight different modalities based on their relevance
- **Multi-Task Learning**: Combines regression with auxiliary classification
- **Advanced Loss Functions**: Focal Huber loss, Log-Cosh loss, and contrastive learning
- **Robust to Missing Images**: Handles cases where product images are unavailable



### Missing Image Handling

Three policies available:
- `zero`: Use zero-filled dummy image
- `text_only`: Skip image processing entirely
- `drop`: Remove samples with missing images

### Mixed Precision Training

Enabled by default via `FP16=true` for faster training and lower memory usage.

### Early Stopping

Training stops automatically if validation SMAPE doesn't improve for `EARLY_STOP_ROUNDS` epochs.

## Tips for Best Results

1. **Data Quality**: Ensure clean, descriptive text and high-quality images
2. **Hyperparameter Tuning**: Experiment with loss weights (`ALPHA_*` parameters)
3. **Batch Size**: Increase if GPU memory allows (better convergence)
4. **Learning Rate**: Lower for fine-tuning pretrained models
5. **Image Policy**: Use `drop` if most samples have images, `zero` otherwise

## Troubleshooting

### Out of Memory
- Reduce `BATCH_SIZE`
- Enable gradient accumulation with `GRAD_ACCUM=2`
- Use a smaller CLIP model

### Poor Performance
- Increase `EPOCHS`
- Adjust loss weights
- Try different `LOSS_TYPE` (focal_huber, log_cosh, huber)
- Ensure images are downloading correctly

### Slow Training
- Reduce `MAX_LEN_CLIP` and `MAX_LEN_DISTIL`
- Use fewer workers in DataLoader
- Enable FP16 training


## Acknowledgments

- CLIP: OpenAI's Contrastive Language-Image Pre-training
- DistilBERT: HuggingFace's distilled BERT model
- Amazon ML Challenge 2025 for the dataset
