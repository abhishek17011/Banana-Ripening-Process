from pathlib import Path
import json, joblib, numpy as np
from .features import FEATURE_NAMES, extract_features_from_image
from .classifier import DESCRIPTIONS, RECOMMENDATIONS, STAGES

MODEL_PATH=Path(__file__).resolve().parents[1]/"models"/"banana_ripeness_model.pkl"
NAMES_PATH=Path(__file__).resolve().parents[1]/"models"/"feature_names.json"
def load_model():
 return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
def extract_features(image): return extract_features_from_image(image)[0]
def predict_ripeness(image):
 bundle=load_model()
 if bundle is None: return None
 features,_,_=extract_features_from_image(image); names=bundle.get("feature_names",FEATURE_NAMES)
 if list(names) != FEATURE_NAMES or any(name not in features for name in names): return None
 vector=np.array([[features[n] for n in names]],dtype=float)
 if not np.isfinite(vector).all(): return None
 model=bundle["model"]
 if not hasattr(model,"predict_proba") or not hasattr(model,"classes_"): return None
 label=int(model.predict(vector)[0]); probs=model.predict_proba(vector)[0]; classes=model.classes_
 if set(classes.tolist()) != {0,1,2} or len(probs) != len(classes): return None
 probabilities={STAGES[int(c)]:float(p) for c,p in zip(classes,probs)}
 stage=STAGES[label]
 ordered=sorted(probabilities.values(),reverse=True)
 return {"stage_number":label+1,"stage":stage,"stage_name":stage,"confidence":float(probabilities[stage]),"probabilities":probabilities,"uncertain":len(ordered)>1 and ordered[0]-ordered[1] < 0.10,"features":features,"model_name":bundle.get("model_name","ML model"),"mode":"Trained image-based classification","description":DESCRIPTIONS[stage],"recommendation":RECOMMENDATIONS[stage]}
def predict_proba(image):
 result=predict_ripeness(image); return None if result is None else result["probabilities"]
