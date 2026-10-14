# Dataset documentation

All numbers below were produced by `python -m src.data.prepare` (see `data/processed/stats.json`
and `artifacts/reports/dataset/`). Nothing here is estimated.

## 1. Dataset survey

| Dataset | Language(s) | Size (as published) | Sentiment labels | Images + text? | Access | Used? |
|---|---|---|---|---|---|---|
| **MemoSen** (Hossain et al., LREC 2022) | Bengali, incl. Bengali–English code-mixing and romanised Bengali | 4,417 memes (paper); 4,372 annotations / 4,369 images in the public release | positive / neutral / negative | Yes: image files + transcribed caption | Public Google Drive link in the authors' GitHub repo | **Yes, primary** |
| Memotion 3 (Mishra et al., 2023) | Hindi–English code-mixed (Hinglish) | 10,000 memes; 7,000 / 1,500 / 1,500 | positive / neutral / negative (+ emotion, intensity) | Yes | **Registration required** (DeFactify-2 @ AAAI-23 form) | Not used, but supported via the generic `csv` adapter |
| Memotion 1 (SemEval-2020 Task 8; Sharma et al., 2020) | English | ≈10 k memes | 5-level, collapsible to 3 | Yes | Kaggle mirror; licence not stated on pages checked | Not used: English only |

**Why MemoSen?** It is the only dataset I found that (a) is in an Indian language, (b) has a
3-class sentiment label matching the project brief, (c) ships both images and text, and (d) can
be downloaded without registration. Memotion 3 fits the brief equally well (Hinglish), but the
organisers ask users to register, and that decision belongs to the project owner. The pipeline
already supports it: see §6.

**Relevance caveat.** Bengali is an Indian language (official in West Bengal and Tripura, and
one of the 22 scheduled languages), but the dataset was created by a Bangladeshi research group.
The cultural context of the memes may therefore be Bangladeshi rather than Indian. I did not
verify the collection sources (see the paper). The image file names (`tangaila (…)`,
`bamboo-vaiya (…)`, `KAM (…)`) suggest a small number of source pages.

## 2. Source, licence and provenance

- Paper: E. Hossain, O. Sharif, M. M. Hoque. *MemoSen: A Multimodal Dataset for Sentiment
  Analysis of Memes.* Proc. LREC 2022, pp. 1542–1554. https://aclanthology.org/2022.lrec-1.165
- Data link: from https://github.com/eftekhar-hossain/Multimodal-Sentiment-LREC2022 (Google Drive
  file `12jxFuMz7jtE1kN9fi1ckspgHcKZhkT_B`, `MemoSen.zip`, 587,944,687 bytes,
  SHA-256 `483f39409818f66cbe6e16ea8c5eea7d05b0589e44e959eb1b8e3c00635593af`, downloaded 2026-10-14).
  The provenance is recorded in `data/raw/memosen_provenance.json`.
- **Licence:** none is stated in the repository or on the data link. This project therefore uses
  the data for non-commercial academic research with citation, and **does not redistribute** it.
  Both `data/raw/` and `data/cache/` are git-ignored.
- Release layout: `Dataset/multi-sent.xlsx` (columns `image_name`, `Captions`, `Label_Sentiment`)
  and `Dataset/Memes/` (4,099 `.jpg` + 270 `.png`). There is no official train/val/test split.
- **Size discrepancy:** the paper reports 4,417 memes, but the release contains 4,372 annotation
  rows and 4,369 image files. All counts in this project refer to the release.

## 3. Preparation pipeline (`src/data/prepare.py`)

1. **Load.** Read the xlsx, then build an id from the image file name (extension kept, so ids are
   unique).
2. **Validate labels.** Map lower-cased labels to `positive / neutral / negative`. Unknown labels
   are removed (there were none).
3. **Validate images.** Each image must exist, decode fully (`PIL.verify` plus reload), be in an
   allowed format, have ≤ 40 MP, and have a minimum side ≥ 32 px. PNG transparency is flattened
   to RGB and the EXIF orientation is applied.
4. **Image cache.** Each valid image is re-saved once as an RGB JPEG (quality 95) with its longest
   side at 448 px, under `data/cache/memosen_448/`. The model input is 224 px, so this only avoids
   re-decoding multi-megapixel files every epoch. The Grad-CAM gallery and the app use the
   originals.
5. **Text normalisation** (`src/data/text_processing.py`):
   - Applied: Unicode NFC; control/format characters removed **except ZWJ/ZWNJ** (needed for
     Bengali/Devanagari conjuncts); URLs → `URL`; @mentions → `@USER`; runs of ≥ 4 identical
     characters capped at 3 (`!!!!!!` → `!!!`); whitespace collapsed.
   - **Not** applied: lower-casing (XLM-R is cased), emoji removal, stop-word removal, stemming,
     transliteration. These would destroy sentiment cues or script information.
6. **Missing text** is kept as an empty string with `has_text=False`. MemoSen has no empty
   captions after normalisation; the synthetic sample contains one to exercise this path.
7. **Script-based language identification.** Per-character Unicode block counts give the dominant
   script, all scripts with ≥ 15 % of letters, and a code-mixing flag. Script alone cannot
   separate English from romanised Bengali/Hindi, and the tool says so rather than guessing.
8. **De-duplication.** Records with the same 256-bit difference hash (dHash) of the image *and*
   the same lower-cased text are collapsed. If duplicates disagreed on the label, all copies would
   be dropped (there were none).
9. **Leakage-safe split.** Union-find links records that share an image hash or an identical
   caption of ≥ 12 characters into groups. Groups are split 70/10/20 (train/val/test), stratified
   by the group's majority label, with seed 42. Every group lands entirely in one split.
10. **Outputs.** `train.csv`, `val.csv`, `test.csv`, `all.csv`, `removed.csv` (with the reason for
    each removal), `stats.json`, and four figures.

## 4. Resulting dataset (MemoSen release, seed 42)

| | Count |
|---|---|
| Annotation rows | 4,372 |
| Removed: image file missing (`KAM (103) (1).{jpg,png}`, `KAM (128) (1).{jpg,png}`) | 4 |
| Removed: exact duplicates (same image + caption + label) | 6 |
| Removed: duplicates with conflicting labels | 0 |
| **Usable memes** | **4,362** |
| Near-duplicate groups with > 1 member (kept, never split across sets) | 16 |
| Groups spanning more than one split | 0 |
| Empty captions | 0 |

**Split sizes and labels**

| Split | Positive | Neutral | Negative | Total |
|---|---|---|---|---|
| Train | 942 | 203 | 1,909 | 3,054 |
| Validation | 134 | 29 | 271 | 434 |
| Test | 269 | 58 | 547 | 874 |
| **Total** | **1,345 (30.8 %)** | **290 (6.6 %)** | **2,727 (62.5 %)** | **4,362** |

The classes are strongly imbalanced, and neutral is rare. This is why **macro-F1** is the primary
metric and why the loss uses (square-root) class weights.

**Scripts in captions** (scripts with ≥ 15 % of letters)

| Scripts | Positive | Neutral | Negative | Total |
|---|---|---|---|---|
| Bengali only | 1,033 | 188 | 1,742 | 2,963 |
| Bengali + Latin (code-mixed) | 226 | 80 | 599 | 905 |
| Latin only (English or romanised Bengali) | 86 | 22 | 386 | 494 |

20.7 % of captions are script-mixed. The share is similar across splits (train 20.5 %, val
21.9 %, test 21.2 %).

**Text length.** 14.8 words on average (median 14, max 106). XLM-R tokens: mean 29.1, median
27, 95th percentile 55, max 235. With `max_length = 96`, truncation affects only a small fraction
of captions.

**Images.** Median 960 × 922 px. The smallest side in the dataset is 177 px.

Figures: `artifacts/reports/dataset/label_distribution.png`, `script_distribution.png`,
`text_length.png`, `image_sizes.png`.

## 5. Limitations and responsible use

- **Size and domain.** About 3 k training memes, apparently from a small number of sources.
  Models trained on them will not transfer reliably to other platforms, time periods or other
  Indian languages.
- **Subjective labels.** Sentiment in memes depends on humour, sarcasm and in-group knowledge.
  The release contains a single label per meme; see the paper for its annotation process.
- **Offensive content.** Some memes contain crude, sexual, political or otherwise offensive
  humour. Content warnings apply to the app's "random test meme" button.
- **Captions are transcriptions.** The text comes from the dataset's transcriptions, not from
  OCR. For a new meme, the user must type the text: there is no OCR in this project.
- **Not for decisions about people.** Do not use the models for content moderation, profiling
  or surveillance without human review.

## 6. Using another dataset (e.g. Memotion 3)

1. Register and download the data, then place it under `data/raw/custom/`.
2. In `configs/config.yaml`, set `dataset.name: csv` and fill in `dataset.csv.*`: the CSV path,
   image folder, column names and a `label_map`. I have **not** seen the Memotion 3 files, so no
   column names are assumed.
3. Run `python -m src.data.prepare`, then retrain.

If a dataset provides an official split column, `prepare.py` keeps that split and reports how
many duplicate groups leak across it.

## 7. Synthetic sample (`data/sample/`)

These are 24 hand-written sentences (Bengali, Hindi, Hinglish, English) with programmatically
drawn "face" images, plus one corrupt image, one missing image and one empty caption. They exist
**only** to test the pipeline, the models' plumbing and the UI. Every image is stamped
"SYNTHETIC SAMPLE – NOT REAL DATA". Any metric computed on them is meaningless and must not be
reported.

## 8. Source pages and the source-disjoint split

The MemoSen file names encode the source page (`KAM (12).jpg` → `KAM`). `prepare.py` stores
this in a `source` column, which is used only for analysis and never as a model input.

| Source page | Memes | Positive | Neutral | Negative |
|---|---|---|---|---|
| KAM | 1,050 | 79.9 % | 1.0 % | 19.0 % |
| ovodro_img | 802 | 4.1 % | 12.0 % | 83.9 % |
| bamboo-vaiya | 538 | 5.8 % | 8.2 % | 86.1 % |
| Bangla Thug Life | 403 | 9.7 % | 2.7 % | 87.6 % |
| Halal Memes | 347 | 22.2 % | 18.2 % | 59.7 % |
| tangaila | 344 | 17.7 % | 9.6 % | 72.7 % |
| nurani-memes | 224 | 44.2 % | 2.7 % | 53.1 % |
| bangla-memes | 212 | 33.0 % | 3.8 % | 63.2 % |
| bangla-gag | 161 | 34.8 % | 6.8 % | 58.4 % |
| chosha | 131 | 0.8 % | 0.8 % | 98.5 % |
| direct-offense | 81 | 8.6 % | 0.0 % | 91.4 % |
| bangla-troll | 69 | 46.4 % | 8.7 % | 44.9 % |

Because label mixes differ this much by page, and pages have recognisable styles, a random
split lets models score partly by recognising the page (see `research_results.md` §5).

`configs/source_disjoint.yaml` builds a second split (`data/processed_source_disjoint/`) in which
every page lies in exactly one split. Pages are ranked by the Jensen–Shannon divergence of their
label distribution from the global one; the most typical pages fill test (≥ 20 %), then
validation (≥ 10 %). Near-duplicate memes re-posted by two pages are removed from the later
split (5 memes).

| Split | Pages | Memes | Positive | Neutral | Negative |
|---|---|---|---|---|---|
| Train | KAM, ovodro_img, bamboo-vaiya, chosha, direct-offense | 2,602 | 911 | 152 | 1,539 |
| Validation | bangla-troll, Halal Memes, Bangla Thug Life | 817 | 148 | 80 | 589 |
| Test | bangla-gag, bangla-memes, tangaila, nurani-memes | 938 | 285 | 58 | 595 |

Note: 839 of the 911 positive training memes (92.1 %) in this split come from KAM.
