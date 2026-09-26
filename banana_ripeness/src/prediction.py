from pathlib import Path
import json, joblib, numpy as np
from .features import FEATURE_NAMES, extract_features_from_image
CLASS_LABELS = {
	0: ("Naturally Ripened", "Natural ripening is predicted from the labeled image features."),
	1: ("Chemically/Artificially Ripened", "Chemical/artificial ripening is predicted from the labeled image features."),
}

MODEL_PATH=Path(__file__).resolve().parents[1]/"models"/"banana_ripeness_model.pkl"
NAMES_PATH=Path(__file__).resolve().parents[1]/"models"/"feature_names.json"
def load_model():
 if not MODEL_PATH.exists(): return None
 try:
  bundle=joblib.load(MODEL_PATH)
 except Exception:
  return None
 if not isinstance(bundle,dict) or list(bundle.get("feature_names",[])) != FEATURE_NAMES: return None
 model=bundle.get("model")
 if not hasattr(model,"predict_proba") or set(getattr(model,"classes_",[])) != {0,1}: return None
 return bundle
def extract_features(image): return extract_features_from_image(image)[0]
def predict_ripeness_from_features(features):
 bundle=load_model()
 if bundle is None: return None
 names=bundle.get("feature_names",FEATURE_NAMES)
 if list(names) != FEATURE_NAMES or any(name not in features for name in names): return None
 vector=np.array([[features[n] for n in names]],dtype=float)
 if not np.isfinite(vector).all(): return None
 model=bundle["model"]
 if not hasattr(model,"predict_proba") or not hasattr(model,"classes_"): return None
 label=int(model.predict(vector)[0]); probs=model.predict_proba(vector)[0]; classes=model.classes_
 if set(classes.tolist()) != {0,1} or label not in CLASS_LABELS or len(probs) != len(classes): return None
 stage,description=CLASS_LABELS[label]
 probabilities={CLASS_LABELS[int(class_id)][0]:float(probability) for class_id,probability in zip(classes,probs)}
 ordered=sorted(probabilities.values(),reverse=True)
 return {"classification":"natural" if label == 0 else "chemical","stage":stage,"confidence":float(probabilities[stage]),"probabilities":probabilities,"uncertain":len(ordered)>1 and ordered[0]-ordered[1] < 0.10,"features":features,"model_name":bundle.get("model_name","ML model"),"mode":"Trained image-based classification","description":description,"recommendation":"Image-based estimate only; laboratory confirmation is required."}
def predict_ripeness(image):
 features,_,_=extract_features_from_image(image)
 return predict_ripeness_from_features(features)
def predict_proba(image):
 result=predict_ripeness(image); return None if result is None else result["probabilities"]
