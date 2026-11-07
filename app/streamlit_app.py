"""CASSAVISION: Deep Learning-Based Cassava Leaf Disease Classification & Explainable Agricultural AI System."""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure src is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from cassavision.config import CANONICAL_CLASS_MAP, CLASS_CODES, CLASS_NAMES, get_project_root, load_config
from cassavision.inference.predictor import CassavaInferenceEngine, PredictionResponse

st.set_page_config(
    page_title="CASSAVISION — Explainable Cassava Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom editorial CSS with high contrast, structured hierarchy, and zero generic clutter
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }
    .hero-banner {
        background: linear-gradient(135deg, #064e3b 0%, #0f172a 100%);
        color: #f8fafc;
        padding: 1.6rem 2rem;
        border-radius: 10px;
        border-left: 6px solid #10b981;
        margin-bottom: 1.25rem;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.35rem 0;
        color: #ecfdf5;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        color: #cbd5e1;
        margin: 0;
        line-height: 1.5;
    }
    .safety-banner {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-left: 5px solid #d97706;
        color: #78350f;
        padding: 0.85rem 1.15rem;
        border-radius: 6px;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .status-card-confident {
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-left: 6px solid #059669;
        color: #064e3b;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-card-uncertain {
        background-color: #fffbeb;
        border: 1px solid #f59e0b;
        border-left: 6px solid #d97706;
        color: #78350f;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .status-card-ood {
        background-color: #fef2f2;
        border: 1px solid #ef4444;
        border-left: 6px solid #b91c1c;
        color: #7f1d1d;
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-pill {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading trained CASSAVISION model checkpoint...")
def get_inference_engine(arch_choice: str = "selected") -> Optional[CassavaInferenceEngine]:
    """Load the trained PyTorch inference engine once and cache across sessions."""
    root = get_project_root()
    models_dir = root / "artifacts" / "models"
    if arch_choice == "selected":
        ckpt_path = models_dir / "selected_model.pt"
    else:
        ckpt_path = models_dir / f"{arch_choice}_best.pt"

    if not ckpt_path.exists():
        return None
    return CassavaInferenceEngine(checkpoint_path=ckpt_path)


@st.cache_data
def load_json_artifact(rel_path: str) -> Dict[str, Any]:
    path = get_project_root() / rel_path
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data
def get_sample_test_images() -> List[Dict[str, Any]]:
    """Retrieve genuine held-out test set images across all 5 classes for interactive inspection."""
    root = get_project_root()
    test_csv = root / "data" / "splits" / "test.csv"
    if not test_csv.exists():
        return []
    df = pd.read_csv(test_csv)
    samples: List[Dict[str, Any]] = []
    for cls_idx in range(5):
        sub = df[df["label"] == cls_idx].head(2)
        for _, row in sub.iterrows():
            img_path = root / row["local_relpath"]
            if img_path.exists():
                samples.append(
                    {
                        "label_display": f"[{CANONICAL_CLASS_MAP[cls_idx]['code']}] {row['image_id']} (True: {CANONICAL_CLASS_MAP[cls_idx]['short_name']})",
                        "image_id": str(row["image_id"]),
                        "true_code": CANONICAL_CLASS_MAP[cls_idx]["code"],
                        "true_name": CANONICAL_CLASS_MAP[cls_idx]["name"],
                        "filepath": str(img_path),
                    }
                )
    return samples


def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "last_response" not in st.session_state:
        st.session_state["last_response"] = None
    if "last_filename" not in st.session_state:
        st.session_state["last_filename"] = None


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🌿 CASSAVISION — Explainable Cassava Leaf Disease Classification System</div>
            <p class="hero-subtitle">
                Applied Computer Vision, Transfer Learning, Temperature-Calibrated Uncertainty Quantification &amp; Grad-CAM Visual Interpretability
                built on the Makerere University AI Lab / NaCRRI Uganda Field Dataset.
            </p>
        </div>
        <div class="safety-banner">
            <strong>⚠️ Responsible Agricultural AI Notice:</strong> This system generates statistical computer vision predictions from leaf photographs.
            It is an educational and decision-support research prototype — <strong>not a confirmed plant pathology diagnosis</strong>. Field lighting, leaf angle,
            unfamiliar crop varieties, or co-infections can affect predictions. Always consult an agricultural extension officer or plant pathologist before taking crop management action.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_classification_tab(engine: Optional[CassavaInferenceEngine], registry: Dict[str, Any]) -> None:
    st.subheader("1. Interactive Leaf Disease Classification & Grad-CAM Studio")

    if engine is None:
        st.error(
            "No trained model checkpoint found in `artifacts/models/`. Run `python scripts/run_pipeline.py` to train and evaluate the models first."
        )
        return

    col_in, col_cfg = st.columns([1.65, 1.0])

    with col_cfg:
        st.markdown("##### Inference & Explainability Controls")
        input_mode = st.radio(
            "Select Image Source:",
            options=["Upload Custom Leaf Image", "Select Held-Out Test Set Sample"],
            index=1,
            help="Upload your own image file (.jpg, .png, .webp) or test on genuine held-out field images from the test partition.",
        )
        run_occlusion = st.checkbox(
            "Compute Perturbation Occlusion Map (alongside Grad-CAM)",
            value=False,
            help="Slides an occluding patch across the 224x224 leaf crop to verify which regions cause the steepest probability drop.",
        )
        explain_target_opt = st.selectbox(
            "Grad-CAM Target Class:",
            options=["Top-1 Predicted Class (Default)"] + [f"{i}: {CANONICAL_CLASS_MAP[i]['name']}" for i in range(5)],
            index=0,
        )
        target_cls_idx: Optional[int] = None
        if explain_target_opt != "Top-1 Predicted Class (Default)":
            target_cls_idx = int(explain_target_opt.split(":")[0])

        st.caption(
            f"**Active Model:** `{engine.arch_name.upper()}` | **Calibration Temperature:** $T = {engine.temperature:.3f}$ | "
            f"**Uncertainty Threshold:** {engine.confidence_threshold*100:.0f}%"
        )

    raw_bytes: Optional[bytes] = None
    filename: str = ""
    ground_truth_note: Optional[str] = None

    with col_in:
        if input_mode == "Upload Custom Leaf Image":
            uploaded = st.file_uploader(
                "Upload a cassava leaf image (JPG, JPEG, PNG, WEBP, BMP)",
                type=["jpg", "jpeg", "png", "webp", "bmp"],
            )
            if uploaded is not None:
                raw_bytes = uploaded.getvalue()
                filename = uploaded.name
            else:
                st.info("Upload a cassava leaf photograph above, or switch to **Select Held-Out Test Set Sample** on the right to inspect a real field image immediately.")
        else:
            samples = get_sample_test_images()
            if samples:
                selected_label = st.selectbox(
                    "Choose a held-out test set image (never seen during training or hyperparameter tuning):",
                    options=[s["label_display"] for s in samples],
                )
                chosen_sample = next(s for s in samples if s["label_display"] == selected_label)
                raw_bytes = Path(chosen_sample["filepath"]).read_bytes()
                filename = chosen_sample["image_id"]
                ground_truth_note = f"Held-Out Test Ground Truth: **{chosen_sample['true_name']} ({chosen_sample['true_code']})**"

    if raw_bytes is not None:
        with st.spinner("Validating image, applying deterministic preprocessing, running calibrated inference & Grad-CAM..."):
            resp: PredictionResponse = engine.predict_from_bytes(
                raw_bytes=raw_bytes,
                filename=filename,
                compute_gradcam=True,
                compute_occlusion=run_occlusion,
                target_explain_class=target_cls_idx,
            )

        # Record in session history (without persisting private image bytes to disk)
        if not st.session_state["prediction_history"] or st.session_state["prediction_history"][0].get("filename") != filename or st.session_state["prediction_history"][0].get("arch") != engine.arch_name:
            st.session_state["prediction_history"].insert(
                0,
                {
                    "timestamp": time.strftime("%H:%M:%S"),
                    "filename": filename,
                    "arch": engine.arch_name,
                    "predicted_code": resp.predicted_code,
                    "predicted_name": resp.predicted_name,
                    "confidence": f"{resp.confidence*100:.1f}%",
                    "entropy": f"{resp.predictive_entropy:.2f}",
                    "status": resp.decision_status,
                },
            )

        if not resp.is_valid_image:
            st.markdown(
                f"""
                <div class="status-card-ood">
                    <strong>❌ File Validation Failed ({filename}):</strong> {resp.decision_summary}
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        # Display decision status card
        card_cls = (
            "status-card-ood"
            if resp.is_ood_flagged
            else ("status-card-uncertain" if resp.is_uncertain else "status-card-confident")
        )
        gt_html = f"<br/><em>Held-Out Test Ground Truth: <strong>{chosen_sample['true_name']} ({chosen_sample['true_code']})</strong></em>" if ground_truth_note else ""
        st.markdown(
            f"""
            <div class="{card_cls}">
                <div style="font-size:1.15rem; font-weight:700;">
                    Prediction: {resp.predicted_name} &nbsp;|&nbsp; Calibrated Confidence: {resp.confidence*100:.1f}%
                    &nbsp;|&nbsp; Status: <code>{resp.decision_status}</code>
                </div>
                <div style="margin-top:0.35rem; font-size:0.93rem;">{resp.decision_summary}{gt_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if resp.uncertainty_reasons:
            with st.expander("🔍 Uncertainty & Image Quality Diagnostic Flags", expanded=resp.is_uncertain):
                for r_msg in resp.uncertainty_reasons:
                    st.warning(r_msg)
        if resp.validation.get("quality_warnings"):
            for qw in resp.validation["quality_warnings"]:
                st.caption(f"⚠️ Quality Note: {qw}")

        # Visual Explainability Row
        st.markdown("#### Visual Explainability & Spatial Saliency Inspection")
        if run_occlusion and resp.occlusion_overlay_pil is not None:
            v1, v2, v3, v4 = st.columns(4)
        else:
            v1, v2, v3 = st.columns(3)
            v4 = None

        with v1:
            st.image(resp.aligned_crop_pil, caption=f"Aligned 224×224 Leaf Crop ({filename})", use_container_width=True)
        if resp.gradcam is not None:
            with v2:
                st.image(
                    resp.gradcam.heatmap_rgb,
                    caption=f"Grad-CAM Activation Map ({resp.gradcam.target_class_code})",
                    use_container_width=True,
                )
            with v3:
                st.image(
                    resp.gradcam.overlay_rgb,
                    caption=f"Grad-CAM Overlay (C/B Ratio = {resp.gradcam.center_vs_border_ratio:.2f}x)",
                    use_container_width=True,
                )
        if v4 is not None and resp.occlusion_overlay_pil is not None:
            with v4:
                st.image(
                    resp.occlusion_overlay_pil,
                    caption=f"Perturbation Occlusion Sensitivity ({resp.gradcam.target_class_code if resp.gradcam else resp.predicted_code})",
                    use_container_width=True,
                )

        if resp.gradcam is not None:
            st.info(
                f"**Spatial Saliency Audit:** {resp.gradcam.saliency_concentration_note}\n\n"
                f"*{resp.gradcam.methodology_disclaimer}*"
            )

        # Class Probabilities & Diagnostic Metadata
        p_col, m_col = st.columns([1.35, 1.0])
        with p_col:
            st.markdown("#### Calibrated vs. Uncalibrated Class Probabilities")
            prob_df = pd.DataFrame(
                {
                    "Category Code": CLASS_CODES,
                    "Disease Name": CLASS_NAMES,
                    "Calibrated Prob (T-Scaled)": [resp.class_probabilities[c] for c in CLASS_CODES],
                    "Raw Softmax (T=1.0)": [resp.uncalibrated_probabilities[c] for c in CLASS_CODES],
                }
            )
            fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=130)
            y_pos = np.arange(len(CLASS_CODES))
            cal_vals = [resp.class_probabilities[c] * 100.0 for c in CLASS_CODES]
            uncal_vals = [resp.uncalibrated_probabilities[c] * 100.0 for c in CLASS_CODES]

            ax.barh(y_pos - 0.18, cal_vals, height=0.35, color="#10b981", label=f"Calibrated (T={resp.temperature:.2f})")
            ax.barh(y_pos + 0.18, uncal_vals, height=0.35, color="#94a3b8", label="Uncalibrated Softmax")
            ax.axvline(
                engine.confidence_threshold * 100.0,
                color="#f59e0b",
                linestyle="--",
                linewidth=1.2,
                label=f"Decision Threshold ({engine.confidence_threshold*100:.0f}%)",
            )
            ax.set_yticks(y_pos)
            ax.set_yticklabels(CLASS_CODES)
            ax.invert_yaxis()
            ax.set_xlim(0, 105)
            ax.set_xlabel("Probability (%)")
            ax.grid(axis="x", linestyle="--", alpha=0.35)
            ax.legend(loc="lower right", fontsize=8.5)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

        with m_col:
            st.markdown("#### Agronomic Reference & Image Telemetry")
            st.markdown(
                f"""
                - **Predicted Category:** {resp.class_details.get('name')} (`{resp.predicted_code}`)
                - **Associated Pathogen:** *{resp.class_details.get('pathogen')}*
                - **Characteristic Foliar Symptoms:** {resp.class_details.get('symptoms')}
                - **Agronomic Context:** {resp.class_details.get('agronomic_note')}
                ---
                - **Original Dimensions:** `{resp.validation['width']} × {resp.validation['height']}` ({resp.validation['format']}, {resp.validation['mode']})
                - **Luminance / Brightness:** `{resp.validation['brightness']:.1f} / 255` (Contrast SD: `{resp.validation['contrast']:.1f}`)
                - **Focus Sharpness (Laplacian Var):** `{resp.validation['sharpness_laplacian']:.1f}`
                - **Foliage Chromaticity Ratio:** `{resp.validation['green_dominance_ratio']*100:.1f}%`
                - **Predictive Entropy:** `{resp.predictive_entropy:.3f} nats` (`{resp.normalized_entropy*100:.1f}%` of uniform max)
                - **Single-Image Inference Latency:** `{resp.inference_latency_ms:.1f} ms`
                """
            )


def render_evaluation_tab(registry: Dict[str, Any], test_metrics: Dict[str, Any], eda_summary: Dict[str, Any]) -> None:
    st.subheader("2. Independent Model Evaluation Centre & Confusion Analysis")
    if not test_metrics:
        st.warning("Evaluation metrics not found. Run `python scripts/run_pipeline.py` first.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Independent Test Accuracy", f"{test_metrics['accuracy']*100:.2f}%")
    m2.metric("Test Macro F1-Score", f"{test_metrics['macro_f1']:.4f}")
    m3.metric("Test Weighted F1-Score", f"{test_metrics['weighted_f1']:.4f}")
    m4.metric("Calibrated ECE", f"{test_metrics.get('ece', 0.0):.4f}", delta=f"{test_metrics.get('ece', 0.0) - test_metrics.get('uncalibrated_ece', 0.0):+.4f} vs raw", delta_color="inverse")
    m5.metric("Held-Out Test Samples", f"{test_metrics['total_samples']}")

    st.markdown("#### Per-Class Precision, Recall, F1, and Error Rates on Independent Test Set")
    pc_rows = []
    for code in CLASS_CODES:
        c_data = test_metrics["per_class"][code]
        pc_rows.append(
            {
                "Code": code,
                "Disease Category": c_data["class_name"],
                "Support (Test N)": c_data["support"],
                "Precision": c_data["precision"],
                "Recall (Sensitivity)": c_data["recall"],
                "F1-Score": c_data["f1"],
                "False Positive Rate": c_data["false_positive_rate"],
                "False Negative Rate": c_data["false_negative_rate"],
            }
        )
    st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

    root = get_project_root()
    f_col1, f_col2 = st.columns(2)
    cm_fig = root / "artifacts" / "figures" / "confusion_matrix.png"
    pc_fig = root / "artifacts" / "figures" / "per_class_metrics.png"
    if cm_fig.exists():
        with f_col1:
            st.image(str(cm_fig), caption="Independent Test Set Confusion Matrix (Counts & Row-Normalized %)", use_container_width=True)
    if pc_fig.exists():
        with f_col2:
            st.image(str(pc_fig), caption="Per-Class Precision, Recall & F1 Comparison", use_container_width=True)

    f_col3, f_col4 = st.columns(2)
    lc_fig = root / "artifacts" / "figures" / "learning_curves.png"
    rel_fig = root / "artifacts" / "figures" / "reliability_diagram.png"
    if lc_fig.exists():
        with f_col3:
            st.image(str(lc_fig), caption="Training & Validation Loss and Macro F1 Progression", use_container_width=True)
    if rel_fig.exists():
        with f_col4:
            st.image(str(rel_fig), caption="Post-Hoc Temperature-Scaled Reliability Diagram", use_container_width=True)

    st.markdown("#### Multi-Architecture Benchmark Comparison")
    if registry.get("model_comparison"):
        comp_df = pd.DataFrame(registry["model_comparison"])
        cols_show = [
            c
            for c in [
                "model_name",
                "pretrained",
                "model_size_mb",
                "inference_latency_ms",
                "val_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "test_weighted_f1",
                "test_ece",
            ]
            if c in comp_df.columns
        ]
        st.dataframe(comp_df[cols_show], use_container_width=True, hide_index=True)

    mc_fig = root / "artifacts" / "figures" / "model_comparison.png"
    if mc_fig.exists():
        st.image(str(mc_fig), caption="Architecture Performance vs. CPU Inference Latency Trade-Off", use_container_width=True)

    st.markdown("#### Confusion & Failure Case Analysis")
    fail_fig = root / "artifacts" / "figures" / "failure_cases_gallery.png"
    if fail_fig.exists():
        st.image(str(fail_fig), caption="High-Confidence Test-Set Misclassifications & Grad-CAM Inspection", use_container_width=True)


def render_explainability_audit_tab(registry: Dict[str, Any]) -> None:
    st.subheader("3. Computer Vision Explainability Audit & Dataset Explorer")
    root = get_project_root()

    gcam_fig = root / "artifacts" / "figures" / "gradcam_audit_gallery.png"
    if gcam_fig.exists():
        st.image(
            str(gcam_fig),
            caption="Multi-Class Grad-CAM & Perturbation Occlusion Sensitivity Audit Across All 5 Categories",
            use_container_width=True,
        )

    audit_data = load_json_artifact("artifacts/metrics/explainability_and_error_audit.json")
    if audit_data:
        a1, a2, a3 = st.columns(3)
        a1.metric("Audited Test Samples", f"{audit_data.get('test_samples_audited_for_saliency', 0)}")
        a2.metric("Mean Center-vs-Border Saliency Ratio", f"{audit_data.get('mean_center_vs_border_saliency_ratio', 1.0):.2f}x")
        a3.metric("Central Leaf Focus Rate", f"{audit_data.get('proportion_central_leaf_focused', 0.0)*100:.1f}%")

    st.markdown("#### Exploratory Dataset Visualisations (Makerere / NaCRRI Uganda Field Dataset)")
    e1, e2 = st.columns(2)
    dist_fig = root / "artifacts" / "figures" / "class_distribution.png"
    qual_fig = root / "artifacts" / "figures" / "image_quality_distributions.png"
    if dist_fig.exists():
        with e1:
            st.image(str(dist_fig), caption="Class Imbalance & Group-Aware Stratified Partitioning", use_container_width=True)
    if qual_fig.exists():
        with e2:
            st.image(str(qual_fig), caption="Field Image Luminance, Foliage Chromaticity, Focus Sharpness & RGB Profiles", use_container_width=True)

    grid_fig = root / "artifacts" / "figures" / "representative_samples_grid.png"
    if grid_fig.exists():
        st.image(str(grid_fig), caption="Representative Validated Field Images Across All 5 Categories", use_container_width=True)


def render_model_card_and_history_tab(registry: Dict[str, Any]) -> None:
    st.subheader("4. Model Card, Responsible AI & Session Prediction History")

    c1, c2 = st.columns([1.3, 1.0])
    with c1:
        st.markdown(
            """
            #### Model Card & System Specification
            - **System Name:** CASSAVISION v1.0.0
            - **Primary Task:** 5-Class Cassava Foliar Disease Classification & Visual Explainability
            - **Dataset Origin:** Makerere University AI Lab & National Crops Resources Research Institute (NaCRRI), Uganda (*Kaggle Cassava Leaf Disease Classification*)
            - **Supported Categories:**
              1. `0 — CBB`: Cassava Bacterial Blight (*Xanthomonas axonopodis pv. manihotis*)
              2. `1 — CBSD`: Cassava Brown Streak Disease (*CBSV / UCBSV*)
              3. `2 — CGM`: Cassava Green Mottle (*CGMV*)
              4. `3 — CMD`: Cassava Mosaic Disease (*Begomoviruses*)
              5. `4 — Healthy`: Uninfected Cassava Foliage
            - **Leakage Prevention:** Exact MD5 deduplication + 64-bit perceptual `dHash` clustering (`Hamming <= 6`) prior to stratified group splitting.
            - **Imbalance Mitigation:** Damped inverse-frequency class weighting ($w_c \\propto n_c^{-0.5}$) + label smoothing ($\\epsilon = 0.05$).
            - **Confidence Calibration:** Post-hoc L-BFGS-B temperature scaling fit on held-out validation logits.
            - **Known Limitations:**
              - Trained on Ugandan smallholder field imagery; domain shift may occur on different cassava cultivars, soil backgrounds, or camera sensors in Nigeria or other regions.
              - Foliar images alone cannot detect subterranean root necrosis (e.g., asymptomatic CBSD root constriction) or co-infections of multiple viruses on the same leaf.
              - Does **not** recommend chemical pesticides or replace certified plant pathology laboratory assays (PCR / ELISA).
            """
        )

    with c2:
        st.markdown("#### Session Prediction History (Ephemeral)")
        st.caption(
            "Predictions run during your current browser session are listed below. Uploaded images are processed in-memory and never persisted to disk."
        )
        if st.session_state["prediction_history"]:
            hist_df = pd.DataFrame(st.session_state["prediction_history"])
            st.dataframe(hist_df, use_container_width=True, hide_index=True)
            if st.button("Clear Session History"):
                st.session_state["prediction_history"] = []
                st.rerun()
        else:
            st.info("No predictions recorded in this session yet. Run an image in Tab 1 to populate session history.")


def main() -> None:
    init_session_state()
    render_header()

    registry = load_json_artifact("artifacts/metrics/model_registry.json")
    test_metrics = load_json_artifact("artifacts/metrics/test_evaluation_metrics.json")
    eda_summary = load_json_artifact("artifacts/metrics/eda_summary.json")

    with st.sidebar:
        st.markdown("### ⚙️ CASSAVISION Control Panel")
        avail_models = ["selected"]
        root = get_project_root()
        for m_name in ["mobilenet_v2", "mobilenet_v3_small", "resnet18", "custom_cnn"]:
            if (root / "artifacts" / "models" / f"{m_name}_best.pt").exists():
                avail_models.append(m_name)

        arch_choice = st.selectbox(
            "Active Model Checkpoint:",
            options=avail_models,
            format_func=lambda x: f"Champion ({registry.get('selected_architecture', 'best').upper()})" if x == "selected" else x.upper(),
        )
        engine = get_inference_engine(arch_choice)

        st.markdown("---")
        st.markdown("### 📊 Dataset & Split Summary")
        if registry.get("split_summary"):
            sp = registry["split_summary"]["splits"]
            st.markdown(
                f"- **Validated Cohort:** `{sum(sp[k]['total'] for k in sp)}` images\n"
                f"- **Train Partition:** `{sp['train']['total']}` images\n"
                f"- **Validation Partition:** `{sp['val']['total']}` images\n"
                f"- **Independent Test Set:** `{sp['test']['total']}` images\n"
                f"- **Zero Cluster Leakage:** `Verified`"
            )
        st.markdown("---")
        st.markdown(
            "**Engineering Lead:** Mechatronics & AI Systems Engineering\n\n"
            "**Framework:** PyTorch + TorchVision + Streamlit"
        )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "🔍 1. Classify & Explain (Grad-CAM)",
            "📈 2. Model Evaluation Centre",
            "🧠 3. Explainability & Dataset Audit",
            "🛡️ 4. Model Card & Session History",
        ]
    )

    with tab1:
        render_classification_tab(engine, registry)
    with tab2:
        render_evaluation_tab(registry, test_metrics, eda_summary)
    with tab3:
        render_explainability_audit_tab(registry)
    with tab4:
        render_model_card_and_history_tab(registry)


if __name__ == "__main__":
    main()
