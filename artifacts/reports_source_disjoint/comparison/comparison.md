# Model comparison (auto-generated from evaluation outputs)

Test examples: 938

| Model | Accuracy (95% CI) | Macro-P | Macro-R | Macro-F1 (95% CI) | Weighted-F1 | F1 positive | F1 neutral | F1 negative | Train time (min) | Latency median (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| Text-only (XLM-R) | 0.5501 [0.518, 0.578] | 0.3381 | 0.3427 | 0.3265 [0.298, 0.355] | 0.5159 | 0.1796 | 0.0805 | 0.7194 | 7.8 | 18.8314 (mps) |
| Image-only (ResNet-50) | 0.5565 [0.526, 0.588] | 0.3708 | 0.3690 | 0.3638 [0.331, 0.398] | 0.5393 | 0.2809 | 0.1053 | 0.7054 | 9.7 | 20.5507 (mps) |
| Multimodal: concat | 0.5235 [0.490, 0.554] | 0.4060 | 0.4312 | 0.3871 [0.356, 0.418] | 0.5408 | 0.3007 | 0.1686 | 0.6920 | 17.2 | 39.3936 (mps) |
| Multimodal: cross-attention | 0.5608 [0.529, 0.593] | 0.3519 | 0.3438 | 0.3380 [0.309, 0.364] | 0.5356 | 0.2494 | 0.0441 | 0.7206 | 21.2 | 43.7509 (mps) |
| Baseline: majority class | 0.6343 [0.603, 0.664] | 0.2114 | 0.3333 | 0.2588 [0.251, 0.266] | 0.4924 | 0.0000 | 0.0000 | 0.7763 | – | – |
| Baseline: source-page prior* | 0.6343 [0.603, 0.664] | 0.2114 | 0.3333 | 0.2588 [0.251, 0.266] | 0.4924 | 0.0000 | 0.0000 | 0.7763 | – | – |

\* The source-page prior uses the meme's source page (metadata, not content) and is a shortcut reference, not a competing model.

## Paired bootstrap: macro-F1 difference (A − B)

| Comparison | Observed Δ | 95% CI | P(A not better) |
|---|---|---|---|
| multimodal_concat vs text | +0.0606 | [0.022, 0.099] | 0.001 |
| multimodal_concat vs image | +0.0232 | [-0.008, 0.055] | 0.080 |
| multimodal_attention vs text | +0.0115 | [-0.025, 0.045] | 0.261 |
| multimodal_attention vs image | -0.0258 | [-0.059, 0.005] | 0.950 |
| text vs baseline_source_prior | +0.0678 | [0.039, 0.096] | 0.000 |
| image vs baseline_source_prior | +0.1051 | [0.072, 0.139] | 0.000 |
| multimodal_concat vs baseline_source_prior | +0.1283 | [0.097, 0.161] | 0.000 |
| multimodal_attention vs baseline_source_prior | +0.0793 | [0.051, 0.105] | 0.000 |
| multimodal_attention vs multimodal_concat | -0.0491 | [-0.079, -0.021] | 1.000 |

## Modality ablation of the multimodal models

| Model | Condition | Accuracy | Macro-F1 | Predictions changed |
|---|---|---|---|---|
| Multimodal: concat | image_neutralised | 0.1226 | 0.0993 | 71.9% |
| Multimodal: concat | text_removed | 0.5373 | 0.3799 | 14.8% |
| Multimodal: cross-attention | image_neutralised | 0.6279 | 0.2609 | 24.6% |
| Multimodal: cross-attention | text_removed | 0.5842 | 0.3124 | 14.6% |
