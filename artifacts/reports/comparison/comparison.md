# Model comparison (auto-generated from evaluation outputs)

Test examples: 874

| Model | Accuracy (95% CI) | Macro-P | Macro-R | Macro-F1 (95% CI) | Weighted-F1 | F1 positive | F1 neutral | F1 negative | Train time (min) | Latency median (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Text-only (XLM-R) | 0.6899 [0.659, 0.722] | 0.5418 | 0.5539 | 0.5449 [0.504, 0.589] | 0.6942 | 0.6365 | 0.2261 | 0.7722 | 9.6 | 18.8276 (mps) |
| Image-only (ResNet-50) | 0.6808 [0.650, 0.714] | 0.5215 | 0.4965 | 0.5034 [0.465, 0.543] | 0.6732 | 0.5622 | 0.1667 | 0.7814 | 10.8 | 19.7530 (mps) |
| Multimodal: concat | 0.7426 [0.713, 0.770] | 0.5948 | 0.5667 | 0.5777 [0.534, 0.621] | 0.7353 | 0.6718 | 0.2424 | 0.8187 | 13.7 | 38.7592 (mps) |
| Multimodal: cross-attention | 0.7426 [0.713, 0.771] | 0.5916 | 0.5885 | 0.5889 [0.547, 0.629] | 0.7434 | 0.6888 | 0.2560 | 0.8219 | 13.7 | 47.8583 (mps) |
| Baseline: majority class | 0.6259 [0.595, 0.659] | 0.2086 | 0.3333 | 0.2566 [0.249, 0.265] | 0.4818 | 0.0000 | 0.0000 | 0.7699 | – | – |
| Baseline: source-page prior* | 0.7826 [0.756, 0.810] | 0.5261 | 0.5295 | 0.5236 [0.503, 0.543] | 0.7516 | 0.7276 | 0.0000 | 0.8431 | – | – |

\* The source-page prior uses the meme's source page (metadata, not content) and is a shortcut reference, not a competing model.

## Paired bootstrap: macro-F1 difference (A − B)

| Comparison | Observed Δ | 95% CI | P(A not better) |
|---|---|---|---|
| multimodal_concat vs text | +0.0327 | [-0.018, 0.084] | 0.097 |
| multimodal_concat vs image | +0.0742 | [0.034, 0.119] | 0.002 |
| multimodal_attention vs text | +0.0440 | [0.004, 0.086] | 0.015 |
| multimodal_attention vs image | +0.0855 | [0.043, 0.132] | 0.000 |
| text vs baseline_source_prior | +0.0214 | [-0.021, 0.067] | 0.159 |
| image vs baseline_source_prior | -0.0201 | [-0.061, 0.021] | 0.852 |
| multimodal_concat vs baseline_source_prior | +0.0541 | [0.014, 0.095] | 0.005 |
| multimodal_attention vs baseline_source_prior | +0.0654 | [0.027, 0.104] | 0.000 |
| multimodal_attention vs multimodal_concat | +0.0113 | [-0.035, 0.058] | 0.318 |

## Modality ablation of the multimodal models

| Model | Condition | Accuracy | Macro-F1 | Predictions changed |
|---|---|---|---|---|
| Multimodal: concat | image_neutralised | 0.6739 | 0.5111 | 23.1% |
| Multimodal: concat | text_removed | 0.7277 | 0.4697 | 14.0% |
| Multimodal: cross-attention | image_neutralised | 0.7311 | 0.4674 | 20.9% |
| Multimodal: cross-attention | text_removed | 0.6751 | 0.3762 | 27.2% |
