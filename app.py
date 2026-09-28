from __future__ import annotations

import io
import os
from datetime import date
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

MODEL_DEFAULT = os.getenv("EARTHLENS_MODEL_ID", "HuggingFaceTB/SmolVLM-500M-Instruct")

st.set_page_config(page_title="EarthLens | Satellite Change Triage", page_icon="🛰️", layout="wide")

st.markdown("""
<style>
:root { --ink:#10231f; --muted:#66766f; --line:#dce7df; --green:#146b50; --lime:#c8f169; }
.block-container {max-width: 1320px; padding-top: 1.5rem;}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, sans-serif; color:var(--ink);}
.stApp {background:linear-gradient(180deg,#f7faf7 0%,#f3f7f3 100%);}
.stApp [data-testid="stSidebar"] {background:#eef4ef; border-right:1px solid #dce7df;}
.hero {padding:1.7rem 2rem; border-radius:22px; background:linear-gradient(120deg,#102d27,#17624b); color:#f4fff6; margin-bottom:1.3rem;}
.hero h1 {font-size:2.45rem; margin:0 0 .35rem 0; color:white; letter-spacing:-.04em;}
.hero p {font-size:1.05rem; margin:0; color:#d5e9dd; max-width:850px;}
.pill {display:inline-block; background:#c8f169; color:#173a2b; padding:.25rem .65rem; border-radius:999px; font-size:.78rem; font-weight:700; margin-bottom:.75rem;}
.card {border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; background:white;}
.step {color:#146b50; font-size:.76rem; font-weight:800; letter-spacing:.12em; text-transform:uppercase; margin-bottom:.2rem;}
.stButton>button[kind="primary"] {background:linear-gradient(100deg,#146b50,#20805e); border:0; border-radius:12px; min-height:3rem; font-weight:700;}
.stButton>button {border-radius:10px;}
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
    focus = st.selectbox("Review focus", ["General visible change", "Built-up expansion", "Water extent", "Vegetation disturbance", "Burn scar / fire impact"])
    register = st.checkbox("Try small-shift image alignment", value=False, help="Optional translation-only ECC alignment. It cannot correct scale, rotation, terrain, or perspective differences.")
    st.caption("Preview mode works without model weights. Image comparison is a visual baseline, not geospatial change detection.")

source = st.radio("Image source", ["Upload image pair", "Built-in synthetic demo"], horizontal=True)
if source == "Upload image pair":
    demo_case = "User imagery"
    st.markdown('<div class="step">01 / Select dated imagery</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### 01 · Earlier image")
        before_file = st.file_uploader("Upload an earlier satellite / aerial image", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"], key="before")
        before_date = st.date_input("Image date", value=date(2024, 1, 1), key="before_date")
    with right:
        st.markdown("#### 02 · Later image")
        after_file = st.file_uploader("Upload a later image of the same area", type=["png", "jpg", "jpeg", "webp", "tif", "tiff"], key="after")
        after_date = st.date_input("Image date", value=date.today(), key="after_date")
    ready = bool(before_file and after_file)
else:
    st.markdown('<div class="step">01 / Explore the controlled demo</div>', unsafe_allow_html=True)
    demo_case = st.selectbox("Synthetic test case", ["New construction", "No visible change", "Lighting shift false alarm"])
    before_file = after_file = None
    before_date, after_date = date(2024, 1, 1), date(2025, 1, 1)
    ready = True
    st.info("Synthetic illustration only—not real satellite data. It shows how EarthLens marks candidate visual differences.")

st.markdown('<div class="step">02 / Set review context</div>', unsafe_allow_html=True)
location = st.text_input("Place or coordinates (optional)", placeholder="e.g., Chennai, India | 13.08 N, 80.27 E")
question = st.text_area("What should EarthLens inspect?", value="Describe the visible differences between these two images. Separate direct visual observations from possible explanations, and mention any image-quality limits.", height=90)


def load_rgb(upload: Any) -> Image.Image:
    im = Image.open(upload)
    # First frame for TIFF/animated formats; convert palette/CMYK/etc consistently.
    if getattr(im, "n_frames", 1) > 1:
        im.seek(0)
    return ImageOps.exif_transpose(im).convert("RGB")


def make_demo_pair(case: str = "New construction") -> tuple[Image.Image, Image.Image, Image.Image]:
    """Create a deterministic, clearly synthetic top-down scene for a no-data demo."""
    rng = np.random.default_rng(17)
    h = w = 512
    noise = rng.normal(0, 7, (h, w, 3))
    base = np.zeros((h, w, 3), dtype=np.float32)
    base[:] = (83, 111, 71)  # muted green land
    base += noise
    before = Image.fromarray(np.uint8(np.clip(base, 0, 255)))
    after = before.copy()
    d1, d2 = ImageDraw.Draw(before), ImageDraw.Draw(after)
    # Stable river and road landmarks make the paired scene easy to compare.
    river = [(0, 95), (90, 120), (180, 105), (280, 148), (390, 132), (512, 170)]
    d1.line(river, fill=(58, 117, 145), width=46)
    d2.line(river, fill=(58, 117, 145), width=46)
    for draw in (d1, d2):
        draw.line([(65, 0), (140, 512)], fill=(184, 174, 143), width=13)
        draw.line([(0, 390), (512, 310)], fill=(184, 174, 143), width=10)
        # Sparse, pale rooftops represent the existing settlement.
        for x, y in [(255, 245), (286, 251), (318, 241), (272, 280), (310, 286)]:
            draw.rectangle((x, y, x + 15, y + 11), fill=(173, 161, 137))
    truth = np.zeros((h, w), dtype=np.uint8)
    if case == "New construction":
        # Known changed rectangle provides ground truth for a controlled algorithm sanity check.
        d2.rectangle((365, 265, 474, 365), fill=(139, 119, 91))
        for x in range(372, 470, 24):
            d2.rectangle((x, 275, x + 15, 291), fill=(192, 179, 147))
            d2.rectangle((x, 306, x + 15, 322), fill=(164, 151, 123))
            d2.rectangle((x, 337, x + 15, 353), fill=(201, 185, 151))
        truth[265:366, 365:475] = 255
    elif case == "Lighting shift false alarm":
        # Deliberate nuisance variation: no actual scene change, only a global brightness shift.
        after = ImageEnhance.Brightness(after).enhance(1.12)
    elif case != "No visible change":
        raise ValueError(f"Unknown synthetic case: {case}")
    return before, after, Image.fromarray(truth)


def evaluate_mask(prediction: Image.Image, truth: Image.Image, valid_mask: Image.Image) -> dict:
    pred = np.asarray(prediction) > 0
    actual = np.asarray(truth) > 0
    valid = np.asarray(valid_mask) > 0
    tp = int(np.logical_and.reduce((pred, actual, valid)).sum())
    fp = int((pred & ~actual & valid).sum())
    fn = int((~pred & actual & valid).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    iou = tp / (tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "iou": iou}


def make_evidence_sheet(analysis: dict) -> bytes:
    def font(size: int):
        for name in ["DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf", "Arial.ttf"]:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    width, height = 1500, 1000
    sheet = Image.new("RGB", (width, height), "#f5f8f4")
    draw = ImageDraw.Draw(sheet)
    draw.rounded_rectangle((36, 30, width - 36, 175), radius=24, fill="#12392e")
    draw.text((68, 52), "EARTHLENS  /  CHANGE REVIEW", fill="#d3ef9b", font=font(20))
    draw.text((68, 87), f"{analysis['before_date']}  to  {analysis['after_date']}", fill="white", font=font(32))
    meta = analysis["location"] or "Location not provided"
    draw.text((68, 135), meta[:100], fill="#d7e7dc", font=font(20))
    images = [analysis["before"], analysis["after"], analysis["overlay"]]
    labels = ["EARLIER IMAGE", "LATER IMAGE", "CANDIDATE DIFFERENCE OVERLAY"]
    panel_w, panel_h, top = 448, 410, 220
    for idx, (image, label) in enumerate(zip(images, labels)):
        x = 36 + idx * (panel_w + 24)
        draw.rounded_rectangle((x, top, x + panel_w, top + panel_h), radius=18, fill="white", outline="#dce7df", width=2)
        draw.text((x + 18, top + 14), label, fill="#146b50", font=font(17))
        thumb = ImageOps.contain(image, (panel_w - 32, panel_h - 62), method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x + (panel_w - thumb.width) // 2, top + 48))
    draw.rounded_rectangle((36, 655, width - 36, 800), radius=18, fill="white", outline="#dce7df", width=2)
    draw.text((62, 680), f"{analysis['changed_pct']:.1f}% candidate pixels  |  Focus: {analysis['focus']}  |  Sensitivity: {analysis['threshold']}", fill="#10231f", font=font(21))
    alignment = analysis["alignment"]
    alignment_text = alignment["message"]
    if alignment["score"] is not None:
        alignment_text += f" (ECC fit {alignment['score']:.3f}; translation {alignment['offset'][0]:.1f}, {alignment['offset'][1]:.1f}px)"
    draw.text((62, 728), alignment_text[:150], fill="#52665d", font=font(18))
    draw.rounded_rectangle((36, 825, width - 36, 970), radius=18, fill="#fff3df", outline="#f1dfbf", width=2)
    draw.text((62, 845), "HUMAN REVIEW REQUIRED", fill="#794d13", font=font(18))
    draw.text((62, 885), "Candidate pixels are a visual baseline, not a verified land-cover change or area measurement.", fill="#554a39", font=font(18))
    draw.text((62, 920), "Misalignment, clouds, shadows, season, sensor, and processing differences can create false signals.", fill="#554a39", font=font(18))
    output = io.BytesIO()
    sheet.save(output, format="PNG", optimize=True)
    return output.getvalue()


def prepare_pair(a: Image.Image, b: Image.Image, max_side: int = 1280) -> tuple[Image.Image, Image.Image]:
    # Resizing to a shared canvas is for visual comparison only, not geographic co-registration.
    target = (max(a.width, b.width), max(a.height, b.height))
    scale = min(1.0, max_side / max(target))
    target = (max(1, int(target[0] * scale)), max(1, int(target[1] * scale)))
    return (
        ImageOps.pad(a, target, method=Image.Resampling.LANCZOS, color=(0, 0, 0)),
        ImageOps.pad(b, target, method=Image.Resampling.LANCZOS, color=(0, 0, 0)),
    )


def difference_products(a: Image.Image, b: Image.Image, threshold_value: int, overlay_opacity: int):
    aa, bb = prepare_pair(a, b)
    alignment = {"applied": False, "score": None, "offset": None, "message": "Not requested"}
    valid = np.ones((aa.height, aa.width), dtype=bool)
    if register:
        try:
            import cv2
            input_gray = cv2.cvtColor(np.asarray(aa), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            template_gray = cv2.cvtColor(np.asarray(bb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-6)
            score, warp = cv2.findTransformECC(template_gray, input_gray, warp, cv2.MOTION_TRANSLATION, criteria)
            dx, dy = float(warp[0, 2]), float(warp[1, 2])
            if score < 0.15 or abs(dx) > aa.width * .2 or abs(dy) > aa.height * .2:
                raise ValueError("Alignment looked unstable; showing the unaligned pair.")
            aligned = cv2.warpAffine(np.asarray(aa), warp, (aa.width, aa.height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            valid_u8 = cv2.warpAffine(np.full((aa.height, aa.width), 255, dtype=np.uint8), warp, (aa.width, aa.height), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT)
            aa = Image.fromarray(aligned)
            valid = valid_u8 > 0
            alignment = {"applied": True, "score": float(score), "offset": (dx, dy), "message": "Translation alignment applied"}
        except Exception as exc:
            alignment["message"] = f"Alignment skipped: {exc}"
    # Per-pixel normalized RGB absolute difference. Candidate mask is deliberately heuristic.
    arr_a = np.asarray(aa, dtype=np.float32)
    arr_b = np.asarray(bb, dtype=np.float32)
    delta = np.abs(arr_a - arr_b).mean(axis=2)
    # Mild blur reduces isolated sensor/compression speckles without spatial registration.
    diff_img = Image.fromarray(np.uint8(np.clip(delta * 3.0, 0, 255))).filter(ImageFilter.GaussianBlur(radius=1.0))
    d = np.asarray(diff_img, dtype=np.float32)
    mask = (d >= threshold_value) & valid
    changed_pct = float(mask.sum() / max(int(valid.sum()), 1) * 100)
    heat = np.zeros((*mask.shape, 3), dtype=np.uint8)
    heat[..., 0] = 255
    heat[..., 1] = 93
    heat[..., 2] = 46
    base = np.asarray(bb, dtype=np.float32)
    alpha = (mask.astype(np.float32) * (overlay_opacity / 100.0))[..., None]
    overlay = np.clip(base * (1 - alpha) + heat * alpha, 0, 255).astype(np.uint8)
    normalized = np.uint8(np.clip(d / max(float(d.max()), 1.0) * 255, 0, 255))
    return (
        aa, bb, Image.fromarray(normalized), Image.fromarray(overlay), changed_pct,
        alignment, Image.fromarray(np.uint8(valid) * 255), Image.fromarray(np.uint8(mask) * 255),
    )


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass
    processor = AutoProcessor.from_pretrained(model_name)
    try:
        model = ModelClass.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
    except Exception:
        model = ModelClass.from_pretrained(model_name, torch_dtype="auto")
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


def ask_vlm_pair(processor, model, before: Image.Image, after: Image.Image, prompt: str) -> str:
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "EARLIER IMAGE"},
        {"type": "image"}, {"type": "text", "text": "LATER IMAGE"},
        {"type": "text", "text": prompt},
    ]}]
    rendered = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=rendered, images=[before, after], return_tensors="pt")
    try:
        inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    except AttributeError:
        pass
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=260, do_sample=False)
    answer_tokens = output[0][inputs["input_ids"].shape[1]:]
    return processor.decode(answer_tokens, skip_special_tokens=True).strip()


def safe_model_analysis(model_name: str, before: Image.Image, after: Image.Image, q: str, review_focus: str):
    try:
        processor, model = load_model(model_name)
        desc_a = ask_vlm(processor, model, before, "Describe only visible land-cover and built features. Do not infer causes.")
        desc_b = ask_vlm(processor, model, after, "Describe only visible land-cover and built features. Do not infer causes.")
        pair_prompt = (f"Review focus: {review_focus}. Compare the two satellite/aerial images in order. Report concrete visible differences, "
                       "separate direct observations from possible explanations, mention uncertainty or image-quality limits, "
                       "and do not claim area measurements. " + q)
        comparison = ask_vlm_pair(processor, model, before, after, pair_prompt)
        return desc_a, desc_b, comparison, None
    except Exception as exc:
        return None, None, None, f"{type(exc).__name__}: {exc}"


analyze = st.button("Analyze image pair", type="primary", width="stretch", disabled=not ready)
if analyze:
    try:
        if source == "Built-in synthetic demo":
            before, after, truth_mask = make_demo_pair(demo_case)
            before_name, after_name = "synthetic-before.png", "synthetic-after.png"
        else:
            before, after = load_rgb(before_file), load_rgb(after_file)
            truth_mask = None
            before_name, after_name = before_file.name, after_file.name
        source_dimensions = (before.size, after.size)
        bef, aft, diff, overlay, changed_pct, alignment, valid_mask, prediction_mask = difference_products(before, after, threshold, opacity)
        synthetic_metrics = evaluate_mask(prediction_mask, truth_mask, valid_mask) if truth_mask is not None else None
        st.session_state["analysis"] = {
            "before": bef, "after": aft, "diff": diff, "overlay": overlay, "valid_mask": valid_mask,
            "prediction_mask": prediction_mask, "truth_mask": truth_mask, "synthetic_metrics": synthetic_metrics,
            "changed_pct": changed_pct, "before_date": before_date.isoformat(), "after_date": after_date.isoformat(),
            "location": location.strip(), "question": question.strip(), "threshold": threshold,
            "before_name": before_name, "after_name": after_name, "source": source, "focus": focus, "alignment": alignment, "demo_case": demo_case,
            "source_dimensions": source_dimensions,
        }
        st.session_state.pop("vlm_result", None)
        if use_vlm:
            with st.spinner(f"Loading {model_id} and analyzing images locally…"):
                result = safe_model_analysis(model_id, bef, aft, question, focus)
            st.session_state["vlm_result"] = result
    except Exception as exc:
        st.error(f"Could not process the image pair: {exc}")

analysis = st.session_state.get("analysis")
if analysis:
    analyzed_before_date = date.fromisoformat(analysis["before_date"])
    analyzed_after_date = date.fromisoformat(analysis["after_date"])
    st.divider()
    st.markdown("### Change review")
    if analyzed_after_date <= analyzed_before_date:
        st.warning("The later image date should be after the earlier image date. Check the dates before interpreting this comparison.")
    st.caption(f"{analysis['before_date']} → {analysis['after_date']}" + (f" · {analysis['location']}" if analysis["location"] else ""))
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate pixels", f"{analysis['changed_pct']:.1f}%", help="Share of pixels above the selected RGB-difference threshold. Not a measured land area or probability.")
    c2.metric("Review focus", analysis["focus"])
    c3.metric("Review status", "Human review")
    st.caption(f"Input sizes: {analysis['source_dimensions'][0][0]} × {analysis['source_dimensions'][0][1]}  →  {analysis['source_dimensions'][1][0]} × {analysis['source_dimensions'][1][1]}  |  Analysis canvas: {analysis['before'].width} × {analysis['before'].height}")
    if analysis["synthetic_metrics"] is not None:
        scores = analysis["synthetic_metrics"]
        st.markdown(f"**Controlled demo check · {analysis['demo_case']}** <span class='smallmuted'>synthetic ground truth only; not real-world performance</span>", unsafe_allow_html=True)
        sm1, sm2, sm3 = st.columns(3)
        sm1.metric("Precision", "n/a" if scores["precision"] is None else f"{scores['precision']:.2f}")
        sm2.metric("Recall", "n/a" if scores["recall"] is None else f"{scores['recall']:.2f}")
        sm3.metric("IoU", "n/a" if scores["iou"] is None else f"{scores['iou']:.2f}")
    tab1, tab2, tab3, tab4 = st.tabs(["Image pair", "Change evidence", "Model review", "Quality checks"])
    with tab1:
        x, y = st.columns(2)
        x.image(analysis["before"], caption=f"Earlier · {analysis['before_date']}", width="stretch")
        y.image(analysis["after"], caption=f"Later · {analysis['after_date']}", width="stretch")
    with tab2:
        if analysis["truth_mask"] is not None:
            x, y, z = st.columns(3)
            x.image(analysis["diff"], caption="RGB difference baseline", width="stretch")
            y.image(analysis["overlay"], caption="Candidate pixels over later image", width="stretch")
            z.image(analysis["truth_mask"], caption="Known synthetic change area", width="stretch")
            scores = analysis["synthetic_metrics"]
            st.caption(f"Controlled mask counts: {scores['tp']:,} true-positive · {scores['fp']:,} false-positive · {scores['fn']:,} false-negative pixels. This tests code behavior on a toy scene only.")
        else:
            x, y = st.columns(2)
            x.image(analysis["diff"], caption="Normalized RGB difference (visual cue)", width="stretch")
            y.image(analysis["overlay"], caption="Candidate pixels over later image", width="stretch")
        st.info("Highlighted pixels are candidates only. Misalignment, clouds, shadows, season, sensor, and processing differences can create false change signals.")
        if analysis["alignment"]["score"] is not None:
            st.caption(f"Translation alignment attempted · ECC fit {analysis['alignment']['score']:.3f} · estimated shift {analysis['alignment']['offset'][0]:.1f}px, {analysis['alignment']['offset'][1]:.1f}px. ECC fit is not a probability or accuracy score.")
        elif analysis["alignment"]["message"] != "Not requested":
            st.warning(analysis["alignment"]["message"])
        st.image(analysis["valid_mask"], caption="Pixels included in the difference calculation (black border excluded after alignment)", width=320)
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

    with tab4:
        st.markdown("#### Before interpreting the highlight, check these conditions")
        dims_a, dims_b = analysis["source_dimensions"]
        ratio_a, ratio_b = dims_a[0] / dims_a[1], dims_b[0] / dims_b[1]
        if abs(ratio_a - ratio_b) / max(ratio_a, ratio_b) > .08:
            st.warning("The source images have noticeably different aspect ratios. They were padded to a shared canvas; check crop and framing manually.")
        else:
            st.success("Source aspect ratios are reasonably similar. This does not confirm that the images are geographically registered.")
        if analyzed_after_date <= analyzed_before_date:
            st.error("The later date must be after the earlier date for a time comparison.")
        else:
            st.write(f"Time interval: **{(analyzed_after_date - analyzed_before_date).days} days**. Seasonal and acquisition differences may still matter.")
        st.write(f"Alignment: **{analysis['alignment']['message']}**")
        if analysis["source"] == "Built-in synthetic demo":
            st.info("This pair is procedurally generated for a controlled UI demonstration. It is not satellite data and is not evidence of model accuracy.")
        st.write("This prototype does not read georeferencing metadata, mask clouds, correct terrain, or validate change classes. Treat the result as a triage cue only.")

    report = f"""# EarthLens review card\n\n- **Location:** {analysis['location'] or 'Not provided'}\n- **Review focus:** {analysis['focus']}\n- **Source mode:** {analysis['source']} · {analysis['demo_case']}\n- **Earlier image:** {analysis['before_name']} ({analysis['before_date']})\n- **Later image:** {analysis['after_name']} ({analysis['after_date']})\n- **Input dimensions:** {analysis['source_dimensions'][0]} → {analysis['source_dimensions'][1]}\n- **Candidate pixels above threshold:** {analysis['changed_pct']:.1f}% (visual baseline only; not land area or probability)\n- **RGB difference threshold:** {analysis['threshold']}\n- **Alignment:** {analysis['alignment']['message']}\n- **Question:** {analysis['question']}\n"""
    if analysis["synthetic_metrics"] is not None:
        scores = analysis["synthetic_metrics"]
        report += f"\n## Controlled synthetic check\nPrecision: {scores['precision']} · Recall: {scores['recall']} · IoU: {scores['iou']} · TP: {scores['tp']} · FP: {scores['fp']} · FN: {scores['fn']}\nThis is a toy-scene sanity check, not real-world remote-sensing accuracy.\n"
    report += "\n## Model output\n"
    result = st.session_state.get("vlm_result")
    if result and result[2] and not result[3]:
        report += f"\n{result[2]}\n"
    else:
        report += "\nNo model interpretation included. Review the difference overlay alongside the original images.\n"
    report += "\n## Review caveat\nCandidate pixels may reflect misalignment, clouds, haze, shadows, season, sensor, or processing differences. This prototype does not georegister images, validate a change class, or provide operational alerts. Human review is required.\n"
    d1, d2 = st.columns(2)
    d1.download_button("Download review card (.md)", report.encode("utf-8"), file_name="earthlens-review.md", mime="text/markdown", width="stretch")
    d2.download_button("Download evidence sheet (.png)", make_evidence_sheet(analysis), file_name="earthlens-evidence.png", mime="image/png", width="stretch")
else:
    st.markdown("""
<div class="card"><strong>Start a review</strong><p class="smallmuted">Upload two views of the same place from different dates. The prototype will show the original pair, a candidate-pixel overlay, and (optionally) an open-model interpretation.</p></div>
""", unsafe_allow_html=True)

st.markdown("<br><span class='smallmuted'>EarthLens is an experimental decision-support prototype. Always inspect the source imagery.</span>", unsafe_allow_html=True)
