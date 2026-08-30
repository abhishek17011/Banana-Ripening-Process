# Banana Ripeness Detection using Image Processing + Machine Learning

A simple Streamlit Version 1 application that classifies a banana as green/unripe, naturally ripened, or chemically ripened from an uploaded photo. It preserves the OpenCV image-processing pipeline and uses a transparent image-based fallback when no trained model is available.

## Run locally

```bash
cd banana_ripeness
pip install -r requirements.txt
streamlit run app.py
```

## Dataset and training

Place only real, labelled images in these folders (JPG, JPEG, or PNG):

```text
dataset/green/     dataset/natural/    dataset/chemical/
```

Each class needs at least 10 valid images to train; 50+ per class is strongly recommended. The trainer ignores corrupted files, reports class distribution and imbalance, uses stratified train/validation/test splits, and never uses the held-out test split for selection.

```bash
python train_model.py
streamlit run app.py
```

Training saves the real extracted feature dataset to `outputs/banana_features.csv`, validation model comparison to `outputs/model_comparison.csv`, held-out evaluation to `outputs/classification_report.txt`, a confusion matrix image, and the chosen model under `models/`. No model or accuracy is generated until a real dataset is supplied.

The trainer compares Random Forest and scaled SVM on validation data, then evaluates the selected model once on the held-out test split. Streamlit automatically loads `models/banana_ripeness_model.pkl` only when it contains the three new classes; otherwise it uses the image-based estimator and reports that model confidence is unavailable. No model or accuracy is generated until real labelled images are supplied.

For deployment, install `requirements.txt`, include the trained `models/` files, and deploy with the Streamlit entry point `app.py`. A feature-based model is a useful baseline; future work can add a CNN/transfer-learning classifier using the same dataset validation and held-out evaluation discipline.

## How it works

The app resizes the image, applies a light 3×3 blur and LAB brightness normalization, then combines HSV, LAB, border-background distance, morphology, connected components, and a GrabCut fallback to estimate the fruit region. Colour and brown-spot analysis are restricted to that mask. Brown spots additionally require local contrast, then are filtered and counted as connected components. Gradient texture and lightweight GLCM-style neighbour features contribute to the image feature vector. Future spectral features can be added separately; no hyperspectral values are generated.

Enable **Show Processing Details** to inspect the segmentation, colour masks, brown-spot mask, gradient output, and the complete feature vector. Tune thresholds in `src/config.py`.

For best results, use a well-lit photo with one banana and a simple background. This is an image-processing estimate, not a food-safety assessment; reliable accuracy across diverse cameras, cultivars, and lighting requires a labelled dataset and trained model.
