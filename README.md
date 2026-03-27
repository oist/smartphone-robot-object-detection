This repo holds the Python code used to retrain the OIST smartphone robot object detector model.
The current training pipeline targets MediaPipe Model Maker and exports MediaPipe-compatible
`.tflite` artifacts for the Android app.

The detector uses three classes:

1. `puck`
2. `robot-front`
3. `robot-back`

The training pipeline now supports two label modes:

1. `three-class` (default): `puck`, `robot-front`, `robot-back`
2. `robot-merged`: `puck`, `robot`

# Current Training Pipeline
The legacy `tflite-model-maker` and `labelImg` flow has been replaced with:

1. Images in `./images`.
2. Annotation with `X-AnyLabeling`.
3. COCO export to `./annotations/coco_detection.json`.
4. Dataset preparation into train/validation/test splits.
5. MediaPipe Model Maker retraining.
6. Packaging the annotated dataset for GitHub Releases.

# Dataset Expectations
The local source-of-truth dataset layout is:

```text
images/
  *.jpg
  *.json      # X-AnyLabeling per-image files
annotations/
  coco_detection.json
```

The COCO `categories` in `annotations/coco_detection.json` must match:

```text
puck
robot-front
robot-back
```

`annotations/coco_detection.json` remains the source-of-truth 3-class export even when training a
`robot-merged` variant. The merge happens during `prepare_dataset.py`.

# Starting From Local Images
The repo is designed to start from local images in [images](/media/HDD/included/code/smartphone-robot/object-detection/images).

The intended local-first flow is:

1. Keep the raw images in `./images`.
2. Open that image directory in `X-AnyLabeling`.
3. Label every object using exactly these class names:
   `puck`, `robot-front`, `robot-back`
4. Let `X-AnyLabeling` save its per-image JSON files in `./images`.
5. Export COCO annotations to `./annotations/coco_detection.json`.
6. Prepare train/validation/test splits and train the model.
7. Package the annotated dataset into a release-ready zip.
8. Upload that zip as a GitHub Release asset.

# X-AnyLabeling Workflow
Use `X-AnyLabeling` against the current local image directory. Its default behavior of saving
per-image JSON files alongside the images is fine. After labeling, export the combined COCO file to
[annotations/coco_detection.json](/media/HDD/included/code/smartphone-robot/object-detection/annotations/coco_detection.json).

Important constraints:

1. Use exactly these labels: `puck`, `robot-front`, `robot-back`
2. Keep image filenames stable between annotation and export
3. Export the combined COCO file to `./annotations/coco_detection.json`

# Dataset Download And Preparation
Create `dataset_release.json` from `dataset_release.example.json` and fill in the real GitHub
Release tag, asset name, and optional SHA256 checksum.

For local training, prepare the MediaPipe dataset splits directly from
[images](/media/HDD/included/code/smartphone-robot/object-detection/images) and
[annotations/coco_detection.json](/media/HDD/included/code/smartphone-robot/object-detection/annotations/coco_detection.json):

```bash
python prepare_dataset.py
```

This will:

1. Read the local images from `./images`.
2. Read the COCO annotations from `./annotations/coco_detection.json`.
3. Split annotated images into `train`, `validation`, and `test` sets.
4. Write MediaPipe-ready splits into `data/prepared/`.

If you want to prepare from a packaged dataset archive instead, you can still use:

```bash
python prepare_dataset.py --source-archive /path/to/object-detection-dataset.zip
```

If neither the local images/annotations nor `--source-archive` are available, the script falls back
to downloading the configured GitHub Release asset.

If the repository is private, set `GITHUB_TOKEN` before using the release-download path.

# Packaging For GitHub Releases
Packaging is part of the intended workflow for this repo. Build the release-ready dataset archive
directly from the local image folder and exported COCO annotations:

```bash
python package_dataset.py --output dataset.zip
```

This creates a zip containing:

```text
dataset-root/
  images/
  labels.json
```

That archive is the expected GitHub Release asset format for later reuse.
`dataset.zip` is the canonical asset name; `object-detection-dataset.zip` is still accepted as a
legacy compatibility name when downloading older releases.

# Training
The primary reusable training entrypoint is:

```bash
python scripts/train_model.py --label-mode three-class
```

Important paths:

1. Training split: `data/prepared/train`
2. Validation split: `data/prepared/validation`
3. Optional test split: `data/prepared/test`
4. Export directory: `exported_model/`

The default model is `mobilenet_multi_avg_i384`, which is one of the currently supported
MediaPipe object detector training architectures.

Reusable training commands:

```bash
python scripts/train_model.py --label-mode three-class
python scripts/train_model.py --label-mode robot-merged
python scripts/train_model.py --label-mode robot-merged -- --epochs 40 --batch-size 4
python scripts/train_model.py --label-mode three-class --build
```

What the wrapper does:

1. Prepares the dataset split with the requested `--label-mode`
2. Runs the matching named Docker Compose service
3. Exports `model.tflite` and, by default, `model_fp16.tflite`
4. Writes `training_summary.json` after export; summary writing is best-effort so it cannot discard a finished model export

You can still call [train.py](/media/HDD/included/code/smartphone-robot/object-detection/train.py)
directly for lower-level control, including `--run-qat` and other training flags.

# Docker
Named Docker Compose services are available for the reusable training modes:

```bash
docker compose up --build mediapipe-model-maker-3class
docker compose up --build mediapipe-model-maker-robot-merged
```

For normal use, prefer the wrapper script above so dataset preparation and training stay aligned.

# Release Publishing
Versioned release assets are prepared under `build/release/<tag>/`. The release scripts rename the
model artifacts so the shipped variant is obvious from the filename.

Prepare release assets for the current 3-class model:

```bash
python scripts/prepare_release_assets.py --tag 2.0.0
```

Publish the GitHub release using those prepared assets:

```bash
python scripts/publish_release.py --tag 2.0.0
```

Release `2.0.0` is documented as the 3-class model release. The repository also supports
`--label-mode robot-merged` for future training runs and future release variants.

The scripts resolve release metadata in this order:

1. `exported_model/training_summary.json` for metrics from fresh training runs
2. `release_inputs/<tag>.json` for release-specific metadata and fallback metrics
3. built-in defaults for the Docker image and version-only release title

That keeps the common release flow short while still allowing older training runs to be published
without editing the scripts.

# DockerHub Publishing
Build and publish the training image to DockerHub:

```bash
python scripts/publish_dockerhub.py --tag 2.0.0
```

The default DockerHub repository is `topher217/smartphone-robot-object-detection`.

# Android Integration
This repo only covers model preparation and retraining. The Android app should consume the exported
`.tflite` model through the MediaPipe object detector runtime.

# Utilities
`downsize.py` can still be used to downsample raw images before annotation or packaging:

```bash
python downsize.py ./images ./downsized
```

`downsize_xml.py` is retained only as a legacy helper for older Pascal VOC workflows and is no
longer part of the supported training path. `validate_coco.py` remains available as an optional
troubleshooting helper, but it is not required in the primary workflow.
