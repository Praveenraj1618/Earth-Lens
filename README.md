# EarthLens

**Satellite-image change triage for human review.** EarthLens compares two dated satellite or aerial images, highlights candidate visual differences, and lets an analyst inspect those signals with a local open-source vision-language model.

> This is a hackathon prototype, not a validated remote-sensing system. Candidate pixels are not confirmed land-cover change or measured area. Misregistration, clouds, haze, shadows, seasonal changes, sensors, and processing can create false signals.

## Prototype features

- Upload before/after image pairs, dates, and optional location.
- Review side-by-side originals, a normalized RGB difference view, and a candidate-pixel overlay.
- Optionally try **translation-only** image alignment with OpenCV ECC. Alignment is best-effort; its fit score is not a confidence or accuracy score.
- Choose a review focus and ask a question. An optional local open model describes each image and receives both images together for a pairwise interpretation.
- Review input-quality checks and download a Markdown review card plus a presentation-ready PNG evidence sheet.
- Run three transparent synthetic checks: known new construction, no scene change, and a brightness-only false alarm.

The synthetic cases provide known toy masks so EarthLens can show precision, recall, and IoU for the simple RGB baseline. They test implementation behavior on generated images only. **They are not evidence of accuracy on satellite imagery.** The lighting-shift case is included to show a concrete failure mode.

## Run locally

Use Python 3.10 or newer.

```bash
git clone https://github.com/Praveenraj1618/Earth-Lens.git
cd Earth-Lens
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Choose **Built-in synthetic demo** for a repeatable walkthrough, or **Upload image pair** to inspect your own data. Use images of the same place and check the acquisition dates and source licenses.

## Optional local vision-language model

The default model is [HuggingFaceTB/SmolVLM-500M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct). Install the optional dependencies:

```bash
pip install -r requirements-vlm.txt
```

Then enable **Use vision-language model** in the sidebar. The initial run downloads weights and needs internet access. Inference runs on the machine hosting the app; it does not send uploaded imagery to a hosted model API. CPU inference can be slow. Set `EARTHLENS_MODEL_ID` to use another compatible Transformers image-text model.

Model descriptions and interpretations can be wrong. Treat them as analyst notes, verify them against the images, and never present generated text as confirmed evidence.

## How the comparison works

1. Images are converted to RGB, aspect-preserving padded to a shared canvas, and resized for display. The app does **not** read georeferencing metadata or perform full geospatial registration.
2. Optional alignment estimates a small translation with OpenCV `findTransformECC`; invalid border pixels are excluded. It cannot correct scale, rotation, terrain relief, perspective, or different map projections. The operator must verify that the result is sensible.
3. The visual baseline computes a lightly smoothed mean absolute RGB pixel difference, then applies the sensitivity threshold. Candidate-pixel share is a percentage of valid display-canvas pixels, not geographic area or a probability.
4. The optional VLM generates separate descriptions and a comparison prompt containing the before/after images. It is asked to separate visible observations from possible explanations.
5. The analyst reviews the images, candidate overlay, input checks, model text, and downloaded evidence sheet.

## Run the tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

GitHub Actions runs syntax checks and the synthetic workflow tests on pushes and pull requests to `main`.

## Demo sequence for judging

1. Select **New construction** and show the known synthetic mask next to the detected overlay and toy precision/recall/IoU.
2. Select **Lighting shift false alarm** and show how a nuisance change triggers the RGB baseline.
3. Upload a real, properly sourced image pair for the Earth-observation demo; use the optional VLM only if it runs reliably on your machine.
4. Explain that translation alignment is limited, compare original images, and download the review card and evidence sheet.

Use imagery you are permitted to display and submit. Record its source, acquisition dates, region, and license in your presentation or submission.

## Project files

```
app.py                       Streamlit application and analysis workflow
requirements.txt             Core UI, image, and alignment dependencies
requirements-vlm.txt         Optional local model dependencies
requirements-dev.txt         Test dependencies
tests/test_prototype.py      Synthetic workflow and alignment smoke tests
.github/workflows/test.yml   GitHub Actions checks
.streamlit/config.toml       EarthLens theme and upload limit
```

## What remains before making real-world claims

- Use co-registered, georeferenced imagery and preserve sensor metadata.
- Evaluate on a documented real satellite dataset with a suitable change-detection baseline and held-out scenes.
- Validate cloud/shadow masking and the chosen change classes; report real precision/recall/IoU by scene.
- Add region selection and evidence-linked model statements only after the underlying spatial grounding is validated.

## Team

Add team members and contributions here before submission.
