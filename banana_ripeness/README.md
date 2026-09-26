# Banana Ripeness Detection using Image Processing + Machine Learning

A Streamlit application that analyzes a banana image with OpenCV and estimates whether it is naturally or chemically/artificially ripened. The app reports image features even when a model is unavailable; it does not infer ripening method from spots or from ripeness-stage dataset labels.

## Run locally

```bash
cd banana_ripeness
pip install -r requirements.txt
streamlit run app.py
```

## Dataset and training

Place only real images with independently verified experimental/ground-truth ripening-method labels in these folders (JPG, JPEG, or PNG):

```text
dataset/natural/     dataset/chemical/
```

Each class needs at least 10 valid images to train; 50+ per class is strongly recommended. The trainer ignores corrupted files, reports class distribution and imbalance, uses stratified train/validation/test splits, and never uses the held-out test split for selection. Existing `ripe`, `unripe`, `turning`, `overripe`, and `spoiled` folders describe visual condition, not ripening method, so they are not used to train this classifier.

```bash
python train_model.py
streamlit run app.py
```

Training saves the real extracted feature dataset to `outputs/banana_features.csv`, validation model comparison to `outputs/model_comparison.csv`, held-out evaluation to `outputs/classification_report.txt`, a confusion matrix image, and the chosen model under `models/`. No model or accuracy is generated until a real dataset is supplied.

The trainer compares Random Forest and scaled SVM on validation data, then evaluates the selected model once on the held-out test split. Streamlit automatically loads `models/banana_ripeness_model.pkl` only when it contains both Natural and Chemical labels and the current feature set. Otherwise, classification is unavailable and the app asks for a correctly labeled dataset/model. No model or accuracy is generated until real ground-truth labels are supplied.

For deployment, install `requirements.txt`, include the trained `models/` files, and deploy with the Streamlit entry point `app.py`. A feature-based model is a useful baseline; future work can add a CNN/transfer-learning classifier using the same dataset validation and held-out evaluation discipline.

## How it works

The app resizes the image, applies a light 3×3 blur and LAB brightness normalization, then combines HSV, LAB, border-background distance, morphology, connected components, and a GrabCut fallback to estimate the fruit region. Colour and brown-spot analysis are restricted to that mask. Brown spots additionally require local contrast, then are filtered and counted as connected components. Gradient texture and lightweight GLCM-style neighbour features contribute to the image feature vector. Future spectral features can be added separately; no hyperspectral values are generated.

Enable **Show Processing Details** to inspect the segmentation, colour masks, brown-spot mask, gradient output, and the complete feature vector. Tune thresholds in `src/config.py`.

For best results, use a well-lit photo with one banana and a simple background. This is an image-processing estimate, not a food-safety assessment; reliable accuracy across diverse cameras, cultivars, and lighting requires a labelled dataset and trained model.
