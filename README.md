# SkillScribe

SkillScribe generates narrative surgical skill assessment reports from video features. It combines SurgVE, a surgical video encoder, with PRISM, an aspect-structured visual aggregator, and a language model adapted with LoRA.

## Files

| File | Description |
| --- | --- |
| `PRISM.py` | Grouped query aggregation, temporal routing, and auxiliary objectives. |
| `train_sur40k.py` | Self-supervised contrastive pre-training of SurgVE. |
| `train_adapter.py` | Stage 1: train PRISM with the language model frozen. |
| `train_lora.py` | Stage 2: adapt the language model with LoRA while keeping PRISM frozen. |
| `eval.py` | Generate reports and compute reference-based text metrics. |
| `eval_fac.py` | Compute AlignScore between generated reports and expert references. |
