# Model Card — CASSAVISION (`v1.0.0`)

## 1. Model Details
- **Model Name:** CASSAVISION Foliar Disease Classifier (`v1.0.0`)
- **System Type:** 5-Class Convolutional Neural Network with Post-Hoc Temperature Calibration, Out-of-Distribution (OOD) Optical Screening, and Grad-CAM / Occlusion Visual Explainability.
- **Evaluated Candidate Architectures:**
  1. `classical_rf_baseline`: 250-tree Balanced Random Forest on 42-D color/vegetation/texture features.
  2. `custom_cnn`: 4-stage BatchNorm/ReLU Convolutional Neural Network trained from scratch.
  3. `mobilenet_v2`: ImageNet-pretrained MobileNetV2 (Inverted Residuals + Linear Bottlenecks).
  4. `mobilenet_v3_small`: ImageNet-pretrained MobileNetV3-Small (Squeeze-and-Excitation + HardSwish).
  5. `resnet18`: ImageNet-pretrained ResNet-18 (Residual Skip Connections).
- **Pretrained Weight Provenance:** Official PyTorch `torchvision.models` ImageNet-1K V1 weights distributed under the BSD-3-Clause open-source license.
- **Input Specification:** RGB leaf image resized to shorter-side $256\text{ px}$, center-cropped to $224 \times 224\text{ px}$, scaled to $[0, 1]$, and normalized with ImageNet channel statistics ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
- **Output Specification:** 5-dimensional logit vector converted via temperature-scaled softmax ($T > 0$) into calibrated class probabilities across the 5 canonical categories.

---

## 2. Intended Use & Intended Users

### 2.1 Primary Purpose
CASSAVISION is an **applied computer vision research prototype and educational decision-support demonstrator** exploring how transfer learning, uncertainty calibration, and visual explainability can assist in screening cassava (*Manihot esculenta*) leaves for four major viral/bacterial conditions and healthy foliage.

### 2.2 Intended Users
- Computer vision engineers, machine learning researchers, and agricultural technology developers studying fine-grained plant pathology classification under severe class imbalance.
- Agricultural students and extension trainees investigating visual symptoms of cassava foliar diseases alongside post-hoc saliency maps.

### 2.3 Supported Categories (Canonical Mapping)
| Index | Code | Full Category Name | Associated Pathogen | Primary Visual Symptoms |
| :---: | :---: | :--- | :--- | :--- |
| `0` | **CBB** | Cassava Bacterial Blight | *Xanthomonas axonopodis* pv. *manihotis* | Angular water-soaked spots, brown necrotic blighting with yellow halos, wilting. |
| `1` | **CBSD** | Cassava Brown Streak Disease | *Cassava brown streak virus* (*CBSV* / *UCBSV*) | Feathery yellow chlorosis along secondary/tertiary veins on mature lower leaves. |
| `2` | **CGM** | Cassava Green Mottle | *Cassava green mottle virus* (*CGMV*) | Faint to distinct green mottling, yellow-green patterns, mild young leaf distortion. |
| `3` | **CMD** | Cassava Mosaic Disease | *Cassava mosaic begomoviruses* (*ACMV* / *EACMV*) | Severe chlorotic mosaic patches, leaf curling, distortion, reduced leaflet area. |
| `4` | **Healthy** | Healthy Foliage | None (Uninfected) | Uniform deep-green palmate lobes, smooth margins, absence of chlorosis or necrosis. |

---

## 3. Dataset Provenance & Partitioning

- **Dataset Origin:** Makerere University AI Lab (Kampala, Uganda) in collaboration with the National Crops Resources Research Institute (NaCRRI), published via the *Kaggle Cassava Leaf Disease Classification* competition (`21,397` total annotated field images).
- **Collection Methodology:** Crowdsourced smartphone photographs captured directly in smallholder farmers' fields across Uganda and annotated by agricultural experts at NaCRRI.
- **Validated Experimental Cohort:** `750` genuine field images ($800 \times 600\text{ px}$ RGB JPEGs, `0` corrupted files) sampled reproducibly (`seed = 42`) with a minimum floor of $80$ images per minority class (`CBB`: $80$, `CBSD`: $80$, `CGM`: $84$, `Healthy`: $90$) while preserving the realistic field majority of `CMD` ($416$ images, $55.47\%$).
- **Leakage Prevention:** Exact MD5 deduplication plus 64-bit difference perceptual hashing (`dHash` with Hamming distance threshold $\le 6$) via Union-Find clustering prior to group-aware stratified splitting (`525` Train / `112` Validation / `113` Held-Out Test).

---

## 4. Training & Calibration Methodology

1. **Two-Phase Progressive Fine-Tuning:**
   - **Phase 1 (Warmup):** Backbone frozen for 2 epochs; newly initialized 5-class linear head trained with AdamW ($\text{lr} = 10^{-3}$).
   - **Phase 2 (Full Fine-Tuning):** Full network unfrozen with discriminative learning rates ($\text{lr}_{\text{backbone}} = 2 \times 10^{-4}$, $\text{lr}_{\text{head}} = 5 \times 10^{-4}$), weight decay $10^{-4}$, gradient clipping ($2.0$), and `CosineAnnealingLR`.
2. **Class Imbalance Mitigation:** Damped inverse-frequency class weights ($w_c \propto n_c^{-0.5}$) combined with label smoothing ($\epsilon = 0.05$) in Cross-Entropy Loss.
3. **Post-Hoc Temperature Scaling:** Scalar temperature $T$ optimized via L-BFGS-B on validation logits to minimize Negative Log-Likelihood and reduce Expected Calibration Error (ECE) without altering top-1 accuracy.

---

## 5. Known Limitations, Risks & Out-of-Scope Uses

### 5.1 Agricultural & Biological Limitations
- **Not a Confirmed Diagnosis:** Visual leaf symptoms can mimic nutrient deficiencies (e.g., nitrogen, magnesium, or zinc deficiency), herbicide drift, mite damage (Cassava Green Mite), or drought stress. Only laboratory molecular assays (PCR, RT-PCR, ELISA) or certified plant pathologists can confirm viral or bacterial etiology.
- **Subterranean Symptoms Invisible:** Cassava Brown Streak Disease (`CBSD`) frequently causes severe dry brown corky necrosis in the tuberous roots even when upper leaves appear mildly symptomatic or healthy. A leaf-only classifier cannot inspect root health.
- **Co-Infections:** Field plants in Sub-Saharan Africa frequently harbor dual infections (e.g., `CMD` + `CBSD` or `CMD` + `CBB`). Because the dataset uses mutually exclusive single-label annotations, the model predicts the visually dominant class rather than multi-label co-infection.
- **Domain Shift:** Images were collected in Uganda. Deployment on different cassava varieties, soil colors, camera sensors, or lighting regimes in Nigeria or West Africa may experience domain shift.

### 5.2 Appropriate vs. Inappropriate Use
- **Appropriate Use:** Educational demonstration, computer vision research benchmarking, preliminary visual triage prior to human expert inspection, and interpretability auditing.
- **Inappropriate / Prohibited Use:**
  - Automatically prescribing or applying chemical pesticides, bactericides, or crop destruction (roguing) without human expert verification.
  - Using the system as a certified commercial phytosanitary quarantine inspection tool.
  - Submitting non-cassava plant species or non-agricultural imagery and treating the forced 5-class output as valid. (While CASSAVISION includes foliage chromaticity and Laplacian blur guardrails, no OOD detector guarantees 100% rejection of unfamiliar inputs.)
