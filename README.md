# EarthLens

**Satellite image change triage for human review.** EarthLens compares two dated images, surfaces candidate visual changes, and helps an analyst inspect them with an open-source vision-language model.

> EarthLens is a prototype. Its image-difference map is a visual cue, not verified land-cover change. Differences can come from misalignment, clouds, haze, shadows, seasons, sensor changes, or image processing. Review the source imagery before acting.

## What it does

- Compare before/after satellite or aerial images with dates and optional location.
- Generate a candidate-change overlay using a simple pixel-difference baseline.
- Use an open vision-language model (SmolVLM by default) to describe each image and compare the pair in one multimodal prompt.
- Export a Markdown review card with metadata, observations, and limitations.
- Include a deterministic **synthetic demo pair** for an immediate walkthrough. It is an illustration, not real satellite imagery.
- Run in Preview mode without downloading a model.

## Run locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Select **Built-in synthetic demo** for an instant example, or **Upload image pair** to use your own images. Add image dates and optional location, adjust the threshold, then select **Analyze image pair**.

## Optional open-source VLM

The default model is `HuggingFaceTB/SmolVLM-500M-Instruct`. Inference uses Transformers and PyTorch. Install the optional model dependencies:

```bash
pip install -r requirements-vlm.txt
```

Then select **Use vision-language model** in the sidebar. The first run downloads model weights from Hugging Face and requires internet access. Model inference may be slow on CPU; a supported GPU can help. If loading fails, switch back to Preview mode. You can select another compatible image-text model by setting `EARTHLENS_MODEL_ID` before starting Streamlit.

This application does not send images to a hosted inference API. In local VLM mode, images are processed on the machine running the app, subject to the selected model's software and model licenses.

## How the prototype works

1. The app loads the image pair onto a shared display canvas while preserving aspect ratio. It does not perform geospatial registration.
2. A normalized RGB absolute-difference map highlights pixels that differ. A threshold suppresses small changes; the overlay marks remaining candidate pixels.
3. The VLM describes each image separately, then receives both images together to compare them. It is prompted to separate visible evidence from uncertain interpretation.
4. The analyst reviews the originals, overlay, and model output, then downloads a Markdown review card.

The pixel baseline does not know geographic coordinates, sensor calibration, or semantic classes. Candidate pixel share is not a measured land area or a probability. The synthetic pair is only for demonstrating the workflow.

## Suggested demo

1. Use the built-in synthetic pair to show the end-to-end interaction, and explicitly identify it as synthetic.
2. For the judged Earth-observation example, load two real images of the same place and dates, ideally with similar season, resolution, and viewing conditions.
3. Compare originals and overlay; explain that highlighted pixels are candidates only.
4. Ask a focused question such as “What visible differences appear between these dated images? Separate direct observations from possible explanations.”
5. Show one failure case (cloud/shadow or seasonal change) and the uncertainty note, then download the review card.

Use imagery that you have permission to redistribute or present, and retain its source, date, and license in your submission.

## Project structure

```
app.py                 Streamlit UI, image comparison, optional VLM, report export
requirements.txt       Core app dependencies
requirements-vlm.txt   Optional local VLM dependencies
README.md              Setup, behavior, limitations, demo guidance
```

## Next steps for the 24-hour final

- Add geospatial metadata and proper image co-registration before comparison.
- Replace the RGB baseline with a validated remote-sensing change-detection method.
- Add region selection, evidence-linked observations, and a small documented evaluation set.
- Benchmark on representative pairs and report measured limits instead of a confidence score.

## Team

Add team members and contributions here before submission.
