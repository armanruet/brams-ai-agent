# Meteor detection model

The BRAMS assistant can use a locally supplied YOLOv11-nano model to identify
candidate meteor echoes in spectrograms. The trained weights are not included
in this repository.

## How the model was prepared

1. **Data:** We used expert annotations from the Radio Meteor Zoo benchmark
   (DOI: `10.48804/6TZP7T`) and obtained the corresponding BRAMS spectrograms.
   Observing days were kept separate across the training, validation and
   held-out test sets.
2. **Preparation:** Spectrograms were processed as RGB images at 640 × 640
   pixels. The lowest 1% of training annotations by contrast signal-to-noise
   ratio were removed; validation and test annotations were left unchanged.
3. **Training:** A pretrained YOLOv11-nano model was fine-tuned for 35 epochs
   on Apple Silicon using the MPS device.
4. **Evaluation:** The held-out test split produced 87.2% mAP@0.5 and 36.4%
   mAP@0.5:0.95. These are detection metrics, not overall accuracy; performance
   on other stations and interference conditions may differ.
5. **Export:** The selected checkpoint was exported to ONNX for local inference.
   The app draws candidate boxes on the spectrogram and estimates echo times from
   their horizontal centers. False positives and missed echoes are possible.

The dataset-building and training scripts are maintained in the separate
`vision_model_work` project and are not included in this repository. The model
manifest in this directory records the original run's settings and metrics.

## Using a local model

Install `requirements-vision.txt`, then place compatible ONNX weights at
`models/yolo11n_brams.onnx` or set `VISION_MODEL_PATH` in your local `.env` to
the weights' location. Restart the app. Without compatible weights, the rest of
the assistant can run, but meteor detection reports that setup is required.
