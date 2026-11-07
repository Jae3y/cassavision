# Career, Resume, Portfolio & Technical Interview Positioning — CASSAVISION

> All metrics, dataset statistics, architectures, and engineering claims in this document are strictly derived from the actual experiments executed in this repository (`data/processed/dataset_provenance.json`, `artifacts/metrics/model_registry.json`, `artifacts/metrics/test_evaluation_metrics.json`, and `artifacts/metrics/explainability_and_error_audit.json`).

---

## 1. Resume-Ready Project Descriptions (Three Tailored Versions)

### Version 1: Computer Vision & Deep Learning Engineer Focus
**CASSAVISION — Explainable Deep Learning System for Cassava Foliar Disease Classification** | *PyTorch, TorchVision, scikit-learn, Pillow, Streamlit*
- Engineered a 5-class agricultural computer vision pipeline on the Makerere University AI Lab / NaCRRI Uganda field dataset (`750` validated $800 \times 600$ RGB images sampled from the `21,397`-image benchmark), implementing 64-bit perceptual `dHash` clustering (`Hamming <= 6`) and MD5 deduplication to guarantee zero cluster leakage across stratified Train (`525`), Validation (`112`), and Test (`113`) splits.
- Benchmarked five architectures—including a 42-feature Random Forest baseline, a 4-stage CNN trained from scratch, and ImageNet-pretrained `MobileNetV2`, `MobileNetV3-Small`, and `ResNet-18`—using 2-phase progressive fine-tuning, discriminative AdamW learning rates, and damped square-root inverse-frequency class weighting ($w_c \propto n_c^{-0.5}$) with label smoothing ($\epsilon = 0.05$).
- Improved independent test Macro F1 from `0.2568` (classical baseline) and `0.4020` (scratch CNN) to **`0.5646` Macro F1**, **`0.6864` Weighted F1**, and **`68.14%` Accuracy** with `MobileNetV2` (`31.06 ms` CPU inference latency, `8.51 MB` footprint).
- Built a custom PyTorch **Grad-CAM** and perturbation **Occlusion Sensitivity** engine auditing spatial saliency across held-out test samples, verifying a **`1.96x` mean center-to-border saliency ratio** (`97.1%` central leaf focus rate).

### Version 2: AI/ML Systems & MLOps Engineering Focus
**CASSAVISION — End-to-End Calibrated Computer Vision & Diagnostic Inference Platform** | *Python, PyTorch, SciPy, Streamlit, Pytest, GitHub Actions*
- Architected a modular, configuration-driven machine learning system (`src/cassavision/`) separating dataset acquisition, automated optical validation, leakage-free splitting, model training, post-hoc calibration, and low-latency inference.
- Implemented automated pre-inference guardrails measuring 2D Laplacian focus variance and Excess-Green foliage chromaticity ratios to flag corrupted, blurred, or out-of-distribution (non-leaf) inputs prior to neural network scoring.
- Integrated post-hoc L-BFGS-B **Temperature Scaling** on validation logits alongside Shannon predictive entropy thresholding, establishing an uncertainty-aware decision policy where correct test predictions averaged `77.16%` confidence versus `60.23%` on misclassifications.
- Delivered a 4-tab interactive Streamlit application and a 10-test automated `pytest` + GitHub Actions CI suite covering edge cases (0-byte files, RGBA transparency, grayscale conversion, OOD rejection, and Grad-CAM tensor hooks).

### Version 3: Agricultural Technology (AgriTech) & Applied AI Focus
**CASSAVISION — Responsible AI Decision-Support Tool for Smallholder Cassava Disease Triage** | *Deep Learning, Transfer Learning, Explainable AI (XAI)*
- Developed an accessible computer vision screening tool targeting four major Sub-Saharan African cassava pathologies—Cassava Mosaic Disease (`CMD`), Cassava Brown Streak Disease (`CBSD`), Cassava Bacterial Blight (`CBB`), and Cassava Green Mottle (`CGM`)—plus healthy foliage.
- Addressed realistic $5.2 : 1$ field prevalence skew between majority `CMD` (`84.3%` test F1) and minority diseases (`CBB`, `CBSD`, `CGM`) via damped cost-sensitive cross-entropy and biologically bounded color augmentations (`hue <= 0.02`) that preserve natural chlorotic symptom coloration.
- Embedded responsible agricultural AI safeguards, pairing every prediction with Grad-CAM visual heatmaps, pathogen profiles, and explicit clinical disclaimers against unsupervised pesticide application.

---

## 2. Technical Skills Matrix (Honest Audit)

### 2.1 Skills Directly Used & Verified in Production Code
- **Deep Learning & Computer Vision:** PyTorch (`torch.nn`, custom autograd tensor hooks), TorchVision (`MobileNetV2`, `MobileNetV3-Small`, `ResNet-18`, custom `CassavaScratchCNN`), transfer learning, 2-phase progressive layer unfreezing, discriminative learning rates, `CosineAnnealingLR`.
- **Image Processing & Optical Diagnostics:** Pillow (`PIL`), NumPy, 2D discrete Laplacian focus/blur variance, Excess Green (`ExG`) chromaticity segmentation, 64-bit difference perceptual hashing (`dHash`), bitwise Hamming distance clustering.
- **Imbalance & Uncertainty Engineering:** Damped inverse-frequency class weighting, label smoothing, post-hoc L-BFGS-B temperature scaling (`scipy.optimize.minimize`), Expected Calibration Error (`ECE`), reliability diagrams, Shannon predictive entropy.
- **Explainable AI (XAI):** Gradient-weighted Class Activation Mapping (**Grad-CAM**), perturbation-based **Occlusion Sensitivity**, quantitative center-vs-border spatial saliency auditing.
- **Classical ML & Data Science:** `scikit-learn` (`RandomForestClassifier`, `train_test_split`, `precision_recall_fscore_support`, `confusion_matrix`), `pandas`, `matplotlib`, `seaborn`.
- **Software Engineering & QA:** Modular package architecture (`pyproject.toml`), YAML configuration management, `Streamlit` UI engineering, `pytest` unit/integration testing, GitHub Actions CI.

### 2.2 Skills Used Experimentally During Benchmarking
- Handcrafted 42-dimensional color/texture feature engineering (RGB/HSV moments, 16-bin green-red histograms, chlorotic ratios, Laplacian percentiles) used to construct the classical Random Forest benchmark.
- Multi-architecture latency vs. parameter-footprint Pareto analysis on CPU hardware.

### 2.3 Skills Considered in Research but Deliberately Not Implemented
- **Vision Transformers (`ViT-B/16`, `Swin`) & Large ConvNeXt Models:** Considered during architectural research but excluded because they require tens of megabytes of weights, exhibit high latency on CPU-only field hardware, and overfit rapidly on $<1,000$ training images compared to inductive-biased inverted-residual networks (`MobileNetV2`).
- **GAN / Diffusion Synthetic Oversampling & Heavy MixUp/CutMix:** Excluded to uphold strict biological fidelity—blending patches of a `CBB` necrotic leaf with a `CMD` chlorotic leaf creates synthetic multi-virus chimeric images when the ground-truth dataset only supports mutually exclusive single-class labels.
- **External Experiment Servers (MLflow / Weights & Biases Cloud Daemons):** Replaced with a deterministic, self-contained local JSON/YAML artifact registry (`artifacts/metrics/model_registry.json`) so the repository runs 100% offline without external tracking accounts.

---

## 3. Personal Website / Portfolio Case Study

### Title
**CASSAVISION: Engineering a Calibrated, Explainable Deep Learning System for Cassava Leaf Disease Classification**

### 1. The Problem & Motivation
Cassava is a vital food security crop across Nigeria and Sub-Saharan Africa, yet viral and bacterial outbreaks (`CMD`, `CBSD`, `CGM`, `CBB`) regularly devastate smallholder harvests. While farmers increasingly have access to entry-level smartphones, black-box image classifiers trained in tutorials often fail in real agricultural settings: they leak near-duplicate images between train and test splits, collapse onto the majority class (`CMD`), output overconfident softmax probabilities on blurry or non-leaf photos, and provide zero visibility into whether predictions are driven by leaf lesions or background soil.

### 2. Engineering the Data Pipeline
Working with the **Makerere University AI Lab / NaCRRI** dataset, I engineered a reproducible pipeline (`src/cassavision/data/`) that validates image integrity, extracts optical telemetry (luminance, contrast, Laplacian sharpness, foliage green ratio), and computes both MD5 byte hashes and 64-bit perceptual `dHash` signatures. Using a Union-Find disjoint-set algorithm, any near-duplicate images (`Hamming <= 6`) are locked into the same cluster before performing a stratified 70/15/15 split (`525` Train / `112` Val / `113` Held-Out Test).

### 3. Addressing Class Imbalance & Biological Augmentation Constraints
In the field dataset, `CMD` makes up over $55\text{–}61\%$ of samples. Standard unweighted training achieves misleadingly acceptable accuracy by ignoring minority classes. I implemented damped square-root inverse-frequency class weights ($w_c \propto n_c^{-0.5}$) with label smoothing ($\epsilon = 0.05$) and bounded training color jitter (`hue <= 0.02`) so healthy green foliage is never artificially shifted into yellow chlorotic mosaic hues.

### 4. Multi-Model Benchmarking & Results
I evaluated five candidate models on the held-out test set:
- **Classical Random Forest (42-D color/texture features):** `59.29%` Accuracy, `0.2568` Macro F1 (`1.36 ms` latency).
- **4-Stage Scratch CNN (`custom_cnn`):** `64.60%` Accuracy, `0.4020` Macro F1 (`18.91 ms` latency).
- **`ResNet-18` (ImageNet-1K):** `63.72%` Accuracy, `0.5015` Macro F1 (`45.66 ms` latency).
- **`MobileNetV3-Small` (ImageNet-1K):** `68.14%` Accuracy, `0.5227` Macro F1 (`18.06 ms` latency).
- **`MobileNetV2` (Champion Selected):** **`68.14%` Accuracy**, **`0.5646` Macro F1**, **`0.6864` Weighted F1** (`31.06 ms` latency, `8.51 MB`).

### 5. Visual Explainability & Uncertainty Guardrails
To ensure predictions are trustworthy, I implemented **Grad-CAM** and **Perturbation Occlusion Sensitivity** alongside post-hoc **Temperature Scaling** and Shannon entropy thresholding. Across 35 audited test images, `MobileNetV2` achieved a **`1.96x` mean center-to-border saliency ratio** (`97.1%` central-leaf focus). Correct test predictions averaged `77.16%` calibrated confidence compared to `60.23%` on errors, allowing the Streamlit interface to automatically flag low-confidence and out-of-distribution inputs.

---

## 4. Technical Interview Preparation Guide

### 4.1 30-Second Elevator Pitch
> "I built **CASSAVISION**, an end-to-end computer vision and explainable AI system that classifies cassava leaf images from the Makerere University / NaCRRI Uganda field dataset into four major disease categories and healthy foliage. Instead of just training a notebook CNN, I engineered perceptual-hash deduplication to prevent data leakage, mitigated severe $5.2:1$ class imbalance using damped inverse-frequency weighting, benchmarked five classical and transfer-learning architectures, calibrated probabilities with temperature scaling, and integrated Grad-CAM and occlusion sensitivity into a tested Streamlit application. Our champion `MobileNetV2` model improved independent test Macro F1 from `0.257` on a classical baseline to `0.565` (`0.686` Weighted F1) at `31 ms` CPU latency."

### 4.2 Two-Minute Project Presentation
1. **Context (20s):** Cassava diseases threaten food security across Sub-Saharan Africa. Field diagnosis from smartphone photos is challenging due to outdoor lighting variation, overlapping yellow-green chlorotic symptoms (`CMD`, `CBSD`, `CGM`), and extreme prevalence skew toward Cassava Mosaic Disease (`CMD`).
2. **Data Engineering & Leakage Prevention (30s):** I built an automated ingestion and validation pipeline over 750 genuine field images ($800 \times 600$ RGB) from the Makerere / NaCRRI dataset. Every image is screened for format integrity, Laplacian focus sharpness, and Excess-Green chromaticity, and grouped via 64-bit `dHash` perceptual hashing (`Hamming <= 6`) so near-duplicates never leak across the 525-train, 112-validation, and 113-test splits.
3. **Modeling & Transfer Learning (40s):** I established two baselines—a 42-feature Balanced Random Forest (`0.257` Test Macro F1) and a 4-stage scratch CNN (`0.402` Test Macro F1)—before evaluating `MobileNetV2`, `MobileNetV3-Small`, and `ResNet-18` using 2-phase progressive fine-tuning (2 epochs of linear head warmup followed by full discriminative-LR fine-tuning with damped class weights $w_c \propto n_c^{-0.5}$). `MobileNetV2` achieved the highest Validation Macro F1 (`0.610`) and Test Macro F1 (`0.565`, Weighted F1 `0.686`, `CMD` F1 `0.843`).
4. **Explainability & Responsible Deployment (30s):** I integrated Grad-CAM, perturbation occlusion maps, post-hoc temperature scaling, and OOD foliage/blur guardrails into a 4-tab Streamlit app verified by 10 automated `pytest` tests.

### 4.3 Defensible Answers to Tough Technical Interview Questions

**Q1: Why is your overall test accuracy 68.14% and Macro F1 0.5646 rather than 95%+? How do you defend that?**
- **Answer:** "First, this dataset consists of noisy, crowdsourced outdoor field images from Uganda—not studio-lit PlantVillage leaves on uniform gray backgrounds. Even top leaderboard ensembles trained on all 21,397 images with heavy GPU TTA typically score in the 88–91% accuracy range because `CMD`, `CBSD`, `CGM`, and `Healthy` share substantial visual overlap and label noise. Second, we trained on a CPU-tractable 525-image training partition with strict leakage prevention and zero test-set peeking. Most importantly, we optimized for **Macro F1** (`0.5646` vs `0.2568` classical baseline) rather than inflating accuracy by over-predicting the majority `CMD` class (`84.3%` F1)."

**Q2: Why did you cap Hue jitter at `0.02` in your training augmentation pipeline?**
- **Answer:** "In general object recognition (e.g., cars or dogs), shifting hue doesn't change the class label. In plant pathology, the primary diagnostic signal distinguishing `Healthy` deep-green foliage from viral `CMD` or `CBSD` is yellow-green chlorosis. Applying a large hue rotation (e.g., `hue=0.2`) turns a healthy green leaf yellow during training while keeping its label as `Healthy`, directly poisoning the decision boundary."

**Q3: Why did `MobileNetV2` outperform `ResNet-18` in your experiments?**
- **Answer:** "`ResNet-18` has $11.18\text{M}$ parameters ($42.6\text{ MB}$), nearly $5\times$ the capacity of `MobileNetV2` ($2.23\text{M}$ parameters, $8.5\text{ MB}$). On a 525-image training set, `ResNet-18`'s high-dimensional last block overfits faster to training-set background textures, whereas `MobileNetV2`'s inverted residual blocks with linear bottlenecks act as an effective structural regularizer while running $32\%$ faster on CPU (`31.06 ms` vs `45.66 ms`)."

**Q4: Does a high Grad-CAM center-to-border ratio (`1.96x`) prove the model understands plant biology?**
- **Answer:** "No. Grad-CAM is a post-hoc correlative visualization of $\text{ReLU}(\sum_k \alpha_k^c A^k)$ at the last convolutional feature map (`7x7` upsampled to `224x224`). It confirms spatially that the model's logit is driven by pixels on the central leaf lamina rather than image corners, but it does not prove causal biological reasoning—for instance, in our error analysis, brown sunburn or dust specks on a healthy leaf lamina still activated `CBB` filters."
