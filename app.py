from __future__ import annotations

import io
import os
from datetime import date
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

MODEL_DEFAULT = os.getenv("EARTHLENS_MODEL_ID", "HuggingFaceTB/SmolVLM-500M-Instruct")

st.set_page_config(page_title="EarthLens | Satellite Change Triage", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
:root { --ink:#10231f; --muted:#66766f; --line:#dce7df; --green:#146b50; --lime:#c8f169; }
.block-container {max-width: 1320px; padding-top: 1.5rem;}
.hero {padding:1.7rem 2rem; border-radius:22px; background:linear-gradient(120deg,#102d27,#17624b); color:#f4fff6; margin-bottom:1.3rem;}
.hero h1 {font-size:2.45rem; margin:0 0 .35rem 0; color:white; letter-spacing:-.04em;}
.hero p {font-size:1.05rem; margin:0; color:#d5e9dd; max-width:850px;}
.pill {display:inline-block; background:#c8f169; color:#173a2b; padding:.25rem .65rem; border-radius:999px; font-size:.78rem; font-weight:700; margin-bottom:.75rem;}
.card {border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; background:white;}
.smallmuted {color:var(--muted); font-size:.9rem;}
[data-testid="stMetric"] {background:#f5f9f5; border:1px solid var(--line); padding:.8rem 1rem; border-radius:14px;}
</style>
<div class="hero"><span class="pill">EARTH OBSERVATION · HUMAN REVIEW</span><h1>EarthLens</h1><p>Compare dated satellite imagery, inspect candidate visual changes, and turn the evidence into a reviewable summary.</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Analysis settings")
    use_vlm = st.toggle("Use vision-language model", value=False, help="Optional local inference with an open model. The first run downloads model weights.")
    model_id = st.text_input("Model ID", value=MODEL_DEFAULT, disabled=not use_vlm)
    threshold = st.slider("Difference sensitivity", min_value=5, max_value=80, value=24, step=1, help="Lower values mark more pixels as candidate change. This is not a probability.")
    opacity = st.slider("Overlay strength", min_value=10, max_value=90, value=55, step=5)
    st.caption("Preview mode works without model weights. Image comparison is a visual baseline, not geospatial change detection.")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("#### 01 · Earlier image")
    before_file = st.file_uploader("Upload an earlier satellite / aerial image", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"], key="before")
    before_date = st.date_input("Image date", value=date(2024, 1, 1), key="before_date")
with right:
    st.markdown("#### 02 · Later image")
    after_file = st.file_uploader("Upload a later image of the same area", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"], key="after")
    after_date = st.date_input("Image date", value=date.today(), key="after_date")

location = st.text_input("Place or coordinates (optional)", placeholder="e.g., Chennai, India · 13.08°N, 80.27°E")
question = st.text_area("What should EarthLens inspect?", value="Describe the visible differences between these two images. Separate direct visual observations from possible explanations, and mention any image-quality limits.", height=90)


def load_rgb(upload: Any) -> Image.Image:
    im = Image.open(upload)
    # First frame for TIFF/animated formats; convert palette/CMYK/etc consistently.
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    return ImageOps.exif_transpose(im).convert("RGB")


def prepare_pair(a: Image.Image, b: Image.Image, max_side: int = 1280) -> tuple[Image.Image, Image.Image]:
    # Resizing to a shared canvas is for visual comparison only, not geographic co-registration.
    target = (max(a.width, b.width), max(a.height, b.height))
    scale = min(1.0, max_side / max(target))
    target = (max(1, int(target[0] * scale)), max(1, int(target[1] * scale)))
    return a.resize(target, Image.Resampling.LANCZOS), b.resize(target, Image.Resampling.LANCZOS)


def difference_products(a: Image.Image, b: Image.Image, threshold_value: int, overlay_opacity: int):
    aa, bb = prepare_pair(a, b)
    # Per-pixel normalized RGB absolute difference. Candidate mask is deliberately heuristic.
    arr_a = np.asarray(aa, dtype=np.float32)
    arr_b = np.asarray(bb, dtype=np.float32)
    delta = np.abs(arr_a - arr_b).mean(axis=2)
    # Mild blur reduces isolated sensor/compression speckles without spatial registration.
    diff_img = Image.fromarray(np.uint8(np.clip(delta * 3.0, 0, 255))).filter(ImageFilter.GaussianBlur(radius=1.0))
    d = np.asarray(diff_img, dtype=np.float32)
    mask = d >= threshold_value
    changed_pct = float(mask.mean() * 100)
    heat = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = 93
    heat[..., 2] = 46
    base = np.asarray(bb, dtype=np.float32)
    alpha = (mask.astype(np.float32) * (overlay_opacity / 100.0))[..., None]
    overlay = np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    normalized = np.uint8(np.clip(d / max(float(d.max()), 1.0) * 255, 0, 255))
    return aa, bb, Image.fromarray(normalized), Image.fromarray(overlay), changed_pct


def load_model(model_name: str):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    processor = AutoProcessor.from_pretrained(model_name)
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    except Exception:
        model = AutoModelForVision2Seq.from_pretrained(model_name, torch_dtype="auto")
    return processor, model


def ask_vlm(processor, model, image: Image.Image, prompt: str) -> str:
    import torch
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=rendered, images=[image], return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    except (StopIteration, AttributeError):
        pass
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=220, do_sample=False)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def safe_model_analysis(model_name: str, before: Image.Image, after: Image.Image, q: str):
    try:
        processor, model = load_model(model_name)
        desc_a = ask_vlm(processor, model, before, "Describe only visible land-cover and built features. Do not infer causes.")
        desc_b = ask_vlm(processor, model, after, "Describe only visible land-cover and built features. Do not infer causes.")
        # Pairwise prompt as a second image-grounded turn: present the later image and explicit observations for comparison.
        pair_prompt = ("Compare this later satellite/aerial image with the earlier image observations below. "
                       "Report concrete visible changes, distinguish observations from hypotheses, state uncertainty, and do not claim area measurements. "
                       f"Earlier image observations: {desc_a}\\nQuestion: {q}")
        comparison = ask_vlm(processor, model, after, pair_prompt)
        return desc_a, desc_b, comparison, None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


analyze = st.button("Analyze image pair", type="primary", use_container_width=True, disabled=not (before_file and after_file))
if analyze:
    try:
        before = load_rgb(before_file)
        after = load_rgb(after_file)
        bef, aft, diff, overlay, changed_pct = difference_products(before, after, threshold, opacity)
        st.session_state["analysis"] = {
            "before": bef, "after": aft, "diff": diff, "overlay": overlay,
            "changed_pct": changed_pct, "before_date": before_date.isoformat(), "after_date": after_date.isoformat(),
            "location": location.strip(), "question": question.strip(), "threshold": threshold,
            "before_name": before_file.name, "after_name": after_file.name,
        }
        st.session_state.pop("vlm_result", None)
        if use_vlm:
            with st.spinner(f"Loading {model_id} and analyzing images locally…"):
                result = safe_model_analysis(model_id, bef, aft, question)
            st.session_state["vlm_result"] = result
    except Exception as exc:
        st.error(f"Could not process the image pair: {exc}")

analysis = st.session_state.get("analysis")
if analysis:
    st.divider()
    st.markdown("### Change review")
    if after_date <= before_date:
        st.warning("The later image date should be after the earlier image date. Check the dates before interpreting this comparison.")
    if bef.width != aft.width or bef.height != aft.height:
        pass
    st.caption(f"{analysis['before_date']} → {analysis['after_date']}" + (f" · {analysis['location']}" if analysis["location"] else ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate pixels", f"{analysis['changed_pct']:.1f}%", help="Share of pixels above the selected RGB-difference threshold. Not a measured land area or probability.")
    c2.metric("Comparison canvas", f"{analysis['before'].width} × {analysis['before'].height}")
    c3.metric("Review status", "Needs human review")
    tab1, tab2, tab3 = st.tabs(["Before / after", "Difference evidence", "Interpretation"])
    with tab1:
        x, y = st.columns(2)
        x.image(analysis["before"], caption=f"Earlier · {analysis['before_date']}", use_container_width=True)
        y.image(analysis["after"], caption=f"Later · {analysis['after_date']}", use_container_width=True)
    with tab2:
        x, y = st.columns(2)
        x.image(analysis["diff"], caption="Normalized RGB difference (visual cue)", use_container_width=True)
        y.image(analysis["overlay"], caption="Candidate pixels over later image", use_container_width=True)
        st.info("Highlighted pixels are candidates only. Misalignment, clouds, shadows, season, sensor, and processing differences can create false change signals.")
    with tab3:
        st.markdown("**Question**: " + (analysis["question"] or "General visible change review"))
        if use_vlm and st.session_state.get("vlm_result"):
            desc_a, desc_b, comparison, err = st.session_state["vlm_result"]
            if err:
                st.error("Local model could not run. The baseline comparison is still available.")
                st.code(err)
                st.caption("Check model compatibility, install torch and transformers, then retry. Preview mode needs no model.")
            else:
                d1, d2 = st.columns(2)
                with d1:
                    st.markdown("**Earlier image: visible observations**")
                    st.write(desc_a)
                with d2:
                    st.markdown("**Later image: visible observations**")
                    st.write(desc_b)
                st.markdown("**Pairwise interpretation**")
                st.write(comparison)
                st.caption("Generated by an open model. Verify each claim against the source images; this output is not a calibrated confidence score.")
        else:
            st.markdown("**Preview mode**")
            st.write("The image-difference baseline has generated candidate regions. Enable the local vision-language model in the sidebar to add image descriptions and a natural-language comparison.")
            st.write("**Suggested analyst check:** verify that the images show the same place, use comparable seasons and viewing conditions, and are aligned before interpreting highlights.")

    report = f"""# EarthLens review card\n\n- **Location:** {analysis['location'] or 'Not provided'}\n- **Earlier image:** {analysis['before_name']} ({analysis['before_date']})\n- **Later image:** {analysis['after_name']} ({analysis['after_date']})\n- **Candidate pixels above threshold:** {analysis['changed_pct']:.1f}% (visual baseline only; not land area or probability)\n- **RGB difference threshold:** {analysis['threshold']}\n- **Question:** {analysis['question']}\n\n## Model output\n"""
    result = st.session_state.get("vlm_result")
    if result and result[2] and not result[3]:
        report += f"\n{result[2]}\n"
    else:
        report += "\nNo model interpretation included. Review the difference overlay alongside the original images.\n"
    report += "\n## Review caveat\nCandidate pixels may reflect misalignment, clouds, haze, shadows, season, sensor, or processing differences. This prototype does not georegister images, validate a change class, or provide operational alerts. Human review is required.\n"
    st.download_button("Download review card (.md)", report.encode("utf-8"), file_name="earthlens-review.md", mime="text/markdown")
else:
    st.markdown("""
<div class="card"><strong>Start a review</strong><p class="smallmuted">Upload two views of the same place from different dates. The prototype will show the original pair, a candidate-pixel overlay, and (optionally) an open-model interpretation.</p></div>
""", unsafe_allow_html=True)

st.markdown("<br><span class='smallmuted'>EarthLens is an experimental decision-support prototype. Always inspect the source imagery.</span>", unsafe_allow_html=True)
