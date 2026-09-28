# EarthLens

**Satellite image change triage for human review.** EarthLens compares two dated images, surfaces candidate visual changes, and helps an analyst inspect them with an open-source vision-language model.

> EarthLens is a prototype. Its image-difference map is a visual cue, not verified land-cover change. Differences can come from misalignment, clouds, haze, shadows, seasons, sensor changes, or image processing. Review the source imagery before acting.

## What it does

- Compare before/after satellite or aerial images with dates and optional location.
- Generate an interpretable candidate-change overlay using a simple pixel-difference baseline.
- Summarize each image and the pair using an open vision-language model (SmolVLM by default), when enabled.
- Export a Markdown review card with metadata, observations, and limitations.
- Run in **Preview mode** without downloading a model, so the interface and baseline overlay can be shown immediately.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a pair of images, enter dates and an optional location, adjust the threshold, then select **Analyze image pair**.

## Optional open-source VLM

The default model is `HuggingFaceTB/SmolVLM-500M-Instruct`. Inference uses Transformers and PyTorch. Install the optional dependencies:

```bash
pip install torch transformers
```

Then select **Use vision-language model** in the sidebar. The first run downloads model weights from Hugging Face and requires internet access. Model inference may be slow on CPU; a supported GPU can help. If loading fails, switch back to Preview mode. You can select another compatible image-text model by setting `EARTHLENS_MODEL_ID` before starting Streamlit.

This application does not send images to a hosted inference API. In local VLM mode, images are processed on the machine running the app, subject to the selected model's software and model licenses.

## How the prototype works

1. The app loads and resizes the image pair to a common display size. It does not perform geospatial registration.
2. A normalized RGB absolute-difference map highlights pixels that differ. A threshold suppresses small changes; the overlay marks remaining candidate pixels.
3. The VLM describes each image and compares the pair. The app asks it to separate visible evidence from uncertain interpretation.
4. The analyst reviews the originals, overlay, and model output, then downloads a Markdown review card.

The pixel baseline is intentionally simple and does not know geographic coordinates, sensor calibration, or semantic classes. It is useful for a prototype demo, not scientific measurement.

## Suggested demo

1. Load two images of the same place from different dates, ideally with similar season, resolution, and viewing conditions.
2. Compare originals and overlay; explain that highlighted pixels are candidates only.
3. Ask a focused question such as “What visible differences appear between these dated images? Separate direct observations from possible explanations.”
4. Show one failure case (cloud/shadow or seasonal change) and the uncertainty note.
5. Download the review card.

Use imagery that you have permission to redistribute or present, and retain its source, date, and license in your submission.

## Project structure

```
app.py                 Streamlit UI, image comparison, optional VLM, report export
requirements.txt       Core dependencies
README.md             Setup, behavior, limitations, demo guidance
```

## Next steps for the 24-hour final

- Add geospatial metadata and proper image co-registration before comparison.
- Replace the RGB baseline with a validated remote-sensing change-detection method.
- Add region selection, evidence-linked observations, and a small documented evaluation set.
- Benchmark on representative pairs and report measured limits instead of a confidence score.

## Team

Add team members and contributions here before submission.
