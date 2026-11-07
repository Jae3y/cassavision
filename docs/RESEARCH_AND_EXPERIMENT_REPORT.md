# Research & Experimental Report — CASSAVISION (`v1.0.0`)

## 1. Agricultural Problem Formulation & Research Context

Cassava (*Manihot esculenta* Crantz) is a staple calorie source and economic livelihood crop for over 500 million people across Sub-Saharan Africa, with Nigeria being the world's largest producer and Uganda serving as a major East African research hub. Smallholder yields are severely constrained by viral and bacterial foliar pathogens that spread through infected stem cuttings and insect vectors (*Bemisia tabaci* whiteflies).

Early visual recognition in farmers' fields is difficult because:
1. Agricultural extension officer-to-farmer ratios in Sub-Saharan Africa often exceed $1 : 3,000$.
2. Foliar symptoms of distinct viral diseases—such as **Cassava Mosaic Disease (`CMD`)**, **Cassava Brown Streak Disease (`CBSD`)**, and **Cassava Green Mottle (`CGM`)**—exhibit overlapping yellow-green chlorotic patterns.
3. Smartphone photographs captured in unconstrained outdoor fields suffer from harsh tropical glare, canopy shadow gradients, motion blur, complex soil/weed backgrounds, and extreme natural class prevalence skew.

### Foundational Literature & Technical Grounding
- **Ramcharan et al. (2017), *Frontiers in Plant Science*:** Demonstrated that transfer learning with deep CNNs on field-captured cassava leaf imagery significantly outperforms handcrafted feature classifiers.
- **Ramcharan et al. (2019), *Frontiers in Plant Science*:** Evaluated mobile-deployed CNNs in Tanzanian fields, highlighting sensitivity to lighting, leaf angle, and symptom severity.
- **Mwebaze et al. (2019), *arXiv:1908.02900 (iCassava 2019 Fine-Grained Visual Categorization Challenge)*:** Established the 5-class benchmark curated by the Makerere University AI Lab and the National Crops Resources Research Institute (NaCRRI) in Uganda (`21,397` annotated images in the expanded Kaggle competition).
- **Selvaraju et al. (2017), *ICCV (Grad-CAM)* & Zeiler & Fergus (2014), *ECCV (Occlusion Sensitivity)*:** Post-hoc spatial attribution techniques essential for auditing whether agricultural CNNs attend to leaf lamina lesions or spurious background soil shortcuts.
- **Guo et al. (2017), *ICML (On Calibration of Modern Neural Networks)*:** Showed that modern deep networks produce overconfident raw softmax scores and established post-hoc **Temperature Scaling** on validation logits to reduce Expected Calibration Error (ECE).

---

## 2. Dataset Provenance, Validation & Exploratory Analysis

### 2.1 Official Dataset & Canonical Class Mapping
We acquired data from the Makerere University AI Lab / NaCRRI Kaggle Cassava Leaf Disease Classification dataset (`Engineer101/cassava-leaf-disease-classification` mirror containing all `21,397` original competition images and `label_num_to_disease_map.json`).

| Class Index | Code | Official Category Name | Full Dataset Count ($N = 21,397$) | Full Dataset Share | Validated Cohort Count ($N = 750$) | Cohort Share | Train ($525$) | Val ($112$) | Held-Out Test ($113$) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `0` | **CBB** | Cassava Bacterial Blight | $1,087$ | $5.08\%$ | $80$ | $10.67\%$ | $56$ | $12$ | $12$ |
| `1` | **CBSD** | Cassava Brown Streak Disease | $2,189$ | $10.23\%$ | $80$ | $10.67\%$ | $56$ | $12$ | $12$ |
| `2` | **CGM** | Cassava Green Mottle | $2,386$ | $11.15\%$ | $84$ | $11.20\%$ | $59$ | $13$ | $12$ |
| `3` | **CMD** | Cassava Mosaic Disease | $13,158$ | $61.49\%$ | $416$ | $55.47\%$ | $291$ | $62$ | $63$ |
| `4` | **Healthy** | Healthy | $2,577$ | $12.04\%$ | $90$ | $12.00\%$ | $63$ | $13$ | $14$ |

### 2.2 Automated Image Validation & Optical Telemetry
Every acquired image passed through `src/cassavision/data/validation.py`:
- **Total Validated Images:** `750` (`0` corrupted or unreadable files).
- **Dimensions & Aspect Ratio:** Uniform $800 \times 600\text{ px}$ RGB JPEGs (aspect ratio $1.3333$, 4:3 smartphone sensor format).
- **Optical & Chromatic Profile Across Classes:**
  - `CBB`: Mean brightness $107.04 \pm 51.30$, mean Laplacian sharpness $1669.23$, foliage green ratio $76.64\%$.
  - `CBSD`: Mean brightness $115.61 \pm 46.83$, mean Laplacian sharpness $1178.76$, foliage green ratio $77.89\%$.
  - `CGM`: Mean brightness $121.19 \pm 47.21$, mean Laplacian sharpness $1754.88$, foliage green ratio $74.58\%$.
  - `CMD`: Mean brightness $116.19 \pm 49.52$, mean Laplacian sharpness $2106.90$, foliage green ratio $79.39\%$.
  - `Healthy`: Mean brightness $111.55 \pm 50.77$, mean Laplacian sharpness $1684.66$, foliage green ratio $74.97\%$.
- **Critical Observation:** Global RGB channel means and overall brightness overlap heavily across all 5 classes (`107.0` to `121.2`). Consequently, any classifier relying solely on global color histograms cannot separate `CBSD`, `CGM`, `CBB`, and `Healthy`. Fine-grained spatial lesion morphology is required.

### 2.3 Duplicate / Leakage Investigation & Group-Aware Splitting
We computed exact MD5 checksums and 64-bit difference perceptual hashes (`dHash`) across all 750 images, grouping any pair with Hamming distance $\le 6$ via a Union-Find disjoint-set structure (`src/cassavision/data/splitting.py`) prior to stratified splitting (`70%` Train / `15%` Val / `15%` Test, `seed = 42`). Zero `image_id`, `md5_hash`, or `cluster_id` overlap exists across `train.csv`, `val.csv`, and `test.csv`.

---

## 3. Experimental Design, Preprocessing & Imbalance Strategy

### 3.1 Preprocessing & Biologically Grounded Augmentation
- **Deterministic Evaluation / Inference Transform:** `Resize(256)` $\to$ `CenterCrop(224, 224)` $\to$ `ToTensor()` $\to$ `Normalize(ImageNet)`.
- **Training Augmentation (`get_train_transforms`):**
  - `RandomResizedCrop(224, scale=(0.75, 1.0))`
  - `RandomHorizontalFlip(p=0.5)` and `RandomVerticalFlip(p=0.2)`
  - `RandomRotation(degrees=20)`
  - `ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)` — **Hue jitter is deliberately capped at $0.02$** because severe hue shifts artificially convert healthy deep-green leaves into yellow-green chlorotic coloration, corrupting ground-truth supervision between `Healthy` and `CMD`/`CBSD`.

### 3.2 Class Imbalance Mitigation
In the training set ($N = 525$), `CMD` ($291$ samples) outnumbers `CBB` ($56$ samples) by $5.20 : 1$. Unweighted Cross-Entropy causes models to over-predict `CMD` (achieving $\sim 55\%$ accuracy with near-zero minority recall), whereas raw inverse-frequency weights ($w_c \propto n_c^{-1}$) destabilize gradients on noisy field labels. We implemented **damped square-root inverse-frequency class weights** ($w_c \propto n_c^{-0.5}$, normalized to unit mean) combined with **label smoothing** ($\epsilon = 0.05$).

---

## 4. Multi-Architecture Benchmark Results

All five models were trained on the exact same `train.csv` ($525$ images), tuned/selected strictly on `val.csv` ($112$ images) by **Validation Macro F1**, calibrated via post-hoc L-BFGS-B temperature scaling on validation logits, and evaluated once on the held-out independent `test.csv` ($113$ images).

| Model Architecture | Pretrained Weights | Params | Size (MB) | Best Epoch | Train Time (s) | CPU Latency (ms/img) | Val Acc | Val Macro F1 | **Test Acc** | **Test Macro F1** | **Test Weighted F1** | Calibrated Test ECE |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `classical_rf_baseline` (42-D Features) | None | — | 2.40 | — | 5.2s | **1.36 ms** | 58.04% | 0.2264 | 59.29% | 0.2568 | 0.4709 | 0.0937 |
| `custom_cnn` (4-Stage Scratch CNN) | None (Kaiming) | 446,661 | 1.70 | 8 | 436.9s | 18.91 ms | 61.61% | 0.3549 | 64.60% | 0.4020 | 0.5889 | **0.0700** |
| **`mobilenet_v2` (Champion Selected)** | **ImageNet-1K V1** | **2,230,277** | **8.51** | **4** | **549.5s** | **31.06 ms** | **73.21%** | **0.6099** | **68.14%** | **0.5646** | **0.6864** | **0.1064** |
| `mobilenet_v3_small` | ImageNet-1K V1 | 1,522,981 | 5.81 | 7 | 228.4s | 18.06 ms | 69.64% | 0.5941 | **68.14%** | 0.5227 | 0.6725 | 0.0709 |
| `resnet18` | ImageNet-1K V1 | 11,179,077 | 42.64 | 6 | 742.1s | 45.66 ms | 68.75% | 0.5826 | 63.72% | 0.5015 | 0.6361 | 0.1105 |

### Key Experimental Findings
1. **Why Accuracy Alone is Misleading:** `classical_rf_baseline` achieves `59.29%` test accuracy simply by predicting `CMD` on most samples, but its **Test Macro F1 is only `0.2568`**, failing on minority diseases.
2. **Value of Transfer Learning:** Moving from the 4-stage `custom_cnn` trained from scratch (`Test Macro F1 = 0.4020`) to ImageNet-pretrained `mobilenet_v2` (`Test Macro F1 = 0.5646`, `Test Weighted F1 = 0.6864`) yields a **+16.26 percentage point gain in Macro F1** and **+9.75 percentage point gain in Weighted F1**. Pretrained intermediate filters immediately capture leaf venation, necrotic spot boundaries, and mosaic chlorosis textures.
3. **Why `mobilenet_v2` Outperformed `resnet18`:** With $525$ training images, `resnet18` ($11.18\text{M}$ parameters, $42.6\text{ MB}$) exhibited faster overfitting on the training set and 47% higher CPU latency ($45.66\text{ ms}$ vs. $31.06\text{ ms}$), whereas `mobilenet_v2` ($2.23\text{M}$ parameters, inverted residual bottlenecks) achieved the highest Validation Macro F1 (`0.6099`) and highest Independent Test Macro F1 (`0.5646`).
4. **Deployment Alternative (`mobilenet_v3_small`):** Matches `mobilenet_v2` on top-1 Test Accuracy (`68.14%`) with 42% faster CPU inference (`18.06 ms`) and lower ECE (`0.0709`), making it an excellent ultra-lightweight edge alternative when minority-class Macro F1 (`0.5227` vs `0.5646`) can be slightly traded for speed.

---

## 5. Per-Class Performance & Confusion Analysis (`mobilenet_v2` on Held-Out Test Set, $N = 113$)

| Class Code | Disease Category | Test Support | Precision | Recall | F1-Score | False Positive Rate | False Negative Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CBB** | Cassava Bacterial Blight | 12 | 0.4706 | 0.6667 | 0.5517 | 0.0891 | 0.3333 |
| **CBSD** | Cassava Brown Streak Disease | 12 | 0.3529 | 0.5000 | 0.4138 | 0.1089 | 0.5000 |
| **CGM** | Cassava Green Mottle | 12 | 0.6667 | 0.6667 | 0.6667 | 0.0396 | 0.3333 |
| **CMD** | Cassava Mosaic Disease | 63 | **0.8793** | **0.8095** | **0.8430** | 0.1400 | 0.1905 |
| **Healthy** | Healthy Foliage | 14 | 0.4444 | 0.2857 | 0.3478 | 0.0505 | 0.7143 |
| **Macro Avg** | Equal Weight Across Classes | 113 | **0.5628** | **0.5857** | **0.5646** | — | — |
| **Weighted Avg** | Support-Weighted Across Classes | 113 | **0.7035** | **0.6814** | **0.6864** | — | — |

### Error & Confusion Diagnosis
1. **`Healthy` $\to$ `CBB` ($5/14$ samples) and `Healthy` $\to$ `CBSD`/`CMD` ($4/14$ samples):** Field-grown "Healthy" cassava leaves in Uganda often contain minor dust, sunburn specks, old insect chew marks, or dry soil background visible between palmate lobes. Because `CBB` is characterized by localized brown necrotic/dry spots, non-pathogenic leaf blemishes or dry background foliage trigger false-positive `CBB` predictions.
2. **`CMD` $\leftrightarrow$ `CBSD` ($8$ `CMD` predicted as `CBSD`) & `CGM` $\to$ `CMD` ($3$ samples):** All three are viral diseases producing yellow-green foliar chlorosis. Mild or early-stage `CMD` without severe leaflet curling resembles the feathery vein chlorosis of `CBSD` or the mottling of `CGM`.
3. **Confidence Separation:** Across the independent test set, the champion model's mean calibrated confidence on **correct predictions is `77.16%`**, whereas its mean confidence on **incorrect predictions drops to `60.23%`**, validating the utility of our confidence/entropy warning policy.

---

## 6. Explainability Audit (`Grad-CAM` & Perturbation Occlusion)

We audited `35` held-out test set predictions using `GradCAMExplainer` on `mobilenet_v2` (`features.18`) and compared the mean activation inside the central $70\%$ leaf window against the outer $15\%$ border margin (`artifacts/metrics/explainability_and_error_audit.json` & `artifacts/figures/gradcam_audit_gallery.png`):
- **Mean Center-to-Border Saliency Ratio:** `1.956x` (Median: `1.868x`).
- **Proportion Central-Leaf Focused ($\text{Ratio} \ge 1.0$):** `97.14%` ($34 / 35$ audited test samples).
- **Insight from Border-Activated Outlier (`3318816619.jpg`, `Healthy`, Ratio = `0.847x`):** Unlike diseased leaves where the model locks onto localized central lesions (`1287334854.jpg` `CBB` ratio = `2.00x`, `734530855.jpg` `CGM` ratio = `2.17x`), confirming that a leaf is `Healthy` requires scanning the broader leaf perimeter to verify the *absence* of lesions—which also makes `Healthy` predictions more susceptible to peripheral soil/background clutter.
