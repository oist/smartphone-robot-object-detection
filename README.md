This repo holds the Python code used to retrain the OIST smartphone robot object detector model.
The current training pipeline targets MediaPipe Model Maker and exports MediaPipe-compatible
`.tflite` artifacts for the Android app.

The detector uses three classes:

1. `puck`
2. `robot-front`
3. `robot-back`

# Current Training Pipeline
The legacy `tflite-model-maker` and `labelImg` flow has been replaced with:

1. Images stored outside git as GitHub Release assets.
2. Annotation with `X-AnyLabeling`.
3. COCO export from `X-AnyLabeling`.
4. Dataset preparation into train/validation/test splits.
5. MediaPipe Model Maker retraining.
6. Export of `.tflite` artifacts with metadata.

# Dataset Expectations
The raw annotation export is expected to be a single COCO dataset zip with this layout:

```text
dataset-root/
  images/
    *.jpg
    *.png
  labels.json
```

The COCO `categories` in `labels.json` must match:

```text
puck
robot-front
robot-back
```

The dataset zip should be published as a GitHub Release asset instead of being committed into git.

# Starting From Local Images
Right now the repo can start from your unlabeled local images in [images](/media/HDD/included/code/smartphone-robot/object-detection/images).

The intended local-first flow is:

1. Keep the raw images in `./images`.
2. Open that image directory in `X-AnyLabeling`.
3. Label every object using exactly these class names:
   `puck`, `robot-front`, `robot-back`
4. Export the finished annotations as a COCO dataset with:
   `images/`
   `labels.json`
5. Validate the export locally.
6. Package the validated COCO dataset into a release-ready zip.
7. Optionally upload that zip as a GitHub Release asset.
8. Prepare train/validation/test splits and train the model.

# X-AnyLabeling Workflow
Use `X-AnyLabeling` against the current local image directory and export a COCO dataset to a
separate folder, for example:

```text
tmp/coco-export/
  images/
  labels.json
```

Important constraints:

1. Use exactly these labels: `puck`, `robot-front`, `robot-back`
2. Keep image filenames stable between annotation and export
3. Ensure the export includes both the copied images and the COCO `labels.json`

# Validate And Package A Finished COCO Export
After you finish labeling in `X-AnyLabeling`, validate the local export:

```bash
python validate_coco.py /path/to/coco-export
```

If validation passes, package it into the release-ready zip format used by this repo:

```bash
python package_dataset.py /path/to/coco-export --output object-detection-dataset.zip
```

That zip can then be uploaded to a GitHub Release or used locally with:

```bash
python prepare_dataset.py --source-archive object-detection-dataset.zip
```

# Dataset Download And Preparation
Create `dataset_release.json` from `dataset_release.example.json` and fill in the real GitHub
Release tag, asset name, and optional SHA256 checksum.

Then prepare the local MediaPipe dataset splits:

```bash
python prepare_dataset.py
```

This will:

1. Download the configured release asset from `oist/smartphone_robot_object_detection`.
2. Extract the COCO dataset into `data/raw/`.
3. Split annotated images into `train`, `validation`, and `test` sets.
4. Write MediaPipe-ready splits into `data/prepared/`.

If the repository is private, set `GITHUB_TOKEN` before running the script.

You can also bypass GitHub Releases and prepare from a local archive:

```bash
python prepare_dataset.py --source-archive /path/to/object-detection-dataset.zip
```

If you have not uploaded a release yet, this local archive path is the correct path to use.

# Training
The default training command assumes the prepared dataset layout created by `prepare_dataset.py`:

```bash
python train.py --export-fp16
```

Important paths:

1. Training split: `data/prepared/train`
2. Validation split: `data/prepared/validation`
3. Optional test split: `data/prepared/test`
4. Export directory: `exported_model/`

The default model is `mobilenet_multi_avg_i384`, which is one of the currently supported
MediaPipe object detector training architectures.

Useful training flags:

```bash
python train.py --epochs 40 --batch-size 4
python train.py --export-fp16
python train.py --run-qat
```

`--export-fp16` exports `model_fp16.tflite` for GPU-oriented deployment.
`--run-qat` adds quantization-aware training and exports `model_int8_qat.tflite` for CPU-oriented deployment.

# Docker
Build the training container and run the default training command:

```bash
docker compose up --build
```

The compose service now uses the MediaPipe training script directly. Prepare the dataset first so
`data/prepared/` exists in the mounted workspace.

# Android Integration
This repo only covers model preparation and retraining. The Android app should consume the exported
`.tflite` model through the MediaPipe object detector runtime.

# Utilities
`downsize.py` can still be used to downsample raw images before annotation or packaging:

```bash
python downsize.py ./images ./downsized
```

`downsize_xml.py` is retained only as a legacy helper for older Pascal VOC workflows and is no
longer part of the supported training path.
