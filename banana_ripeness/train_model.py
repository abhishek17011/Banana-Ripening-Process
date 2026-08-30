from pathlib import Path
import json, warnings
import joblib, matplotlib.pyplot as plt, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from src.dataset import validation_report, CLASS_NAMES
from src.features import FEATURE_NAMES, extract_features_from_path

ROOT=Path(__file__).resolve().parent; OUTPUTS=ROOT/"outputs"; MODELS=ROOT/"models"
def scores(model,x,y):
 p=model.predict(x); precision,recall,f1,_=precision_recall_fscore_support(y,p,average="macro",zero_division=0); return {"Accuracy":accuracy_score(y,p),"Precision":precision,"Recall":recall,"F1 Score":f1}
def main():
 records,invalid,counts,table,errors,warnings_list=validation_report(ROOT/"dataset"); print(table.to_string(index=False))
 for item in invalid: print("Ignored corrupted image:",item)
 if errors:
  for item in errors: print("ERROR:",item)
  raise SystemExit("Training stopped: every class requires at least 10 valid real images.")
 for item in warnings_list: print("WARNING:",item)
 rows=[]
 for record in records:
  try:
   features,_,_=extract_features_from_path(record["filename"]); rows.append({"filename":record["filename"],"label":record["label"],**features})
  except Exception as exc: print("Ignored feature extraction failure:",record["filename"],exc)
 frame=pd.DataFrame(rows); OUTPUTS.mkdir(exist_ok=True); frame.to_csv(OUTPUTS/"banana_features.csv",index=False)
 if frame["label"].value_counts().min()<10: raise SystemExit("Training stopped after extraction: a class has fewer than 10 usable images.")
 x=frame[FEATURE_NAMES]; y=frame["label"]
 # Held-out test is never used during selection; validation chooses the model.
 x_train_val,x_test,y_train_val,y_test=train_test_split(x,y,test_size=.15,stratify=y,random_state=42)
 validation_size=.1765
 x_train,x_val,y_train,y_val=train_test_split(x_train_val,y_train_val,test_size=validation_size,stratify=y_train_val,random_state=42)
 candidates={"Random Forest":RandomForestClassifier(n_estimators=350,max_depth=None,min_samples_split=2,min_samples_leaf=1,class_weight="balanced",random_state=42,n_jobs=-1),"SVM":Pipeline([("scaler",StandardScaler()),("svc",SVC(C=3,gamma="scale",kernel="rbf",probability=True,random_state=42))])}
 comparison=[]
 for name,model in candidates.items():
  model.fit(x_train,y_train); comparison.append({"Model":name,**scores(model,x_val,y_val)})
 comparison_df=pd.DataFrame(comparison); print("\nValidation comparison\n",comparison_df.to_string(index=False)); comparison_df.to_csv(OUTPUTS/"model_comparison.csv",index=False)
 winner=comparison_df.sort_values(["F1 Score","Accuracy"],ascending=False).iloc[0]; model_name=winner["Model"]; best=candidates[model_name]; best.fit(x_train_val,y_train_val)
 test_metrics=scores(best,x_test,y_test); print("\nHeld-out test metrics:",test_metrics)
 report=classification_report(y_test,best.predict(x_test),labels=range(3),target_names=[x.title() for x in CLASS_NAMES],zero_division=0)
 (OUTPUTS/"classification_report.txt").write_text(report+"\n\nTest metrics:\n"+json.dumps(test_metrics,indent=2),encoding="utf-8")
 fig,ax=plt.subplots(figsize=(7,5)); ConfusionMatrixDisplay.from_predictions(y_test,best.predict(x_test),labels=range(3),display_labels=[x.title() for x in CLASS_NAMES],cmap="YlGn",ax=ax,colorbar=False); fig.tight_layout(); fig.savefig(OUTPUTS/"confusion_matrix.png",dpi=150); plt.close(fig)
 if len(frame)>=125:
  cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42); cv_scores=cross_val_score(best,x,y,cv=cv,scoring="accuracy"); print(f"5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
 else: print("Cross-validation skipped: fewer than 125 images.")
 MODELS.mkdir(exist_ok=True); joblib.dump({"model":best,"feature_names":FEATURE_NAMES,"model_name":model_name,"test_metrics":test_metrics},MODELS/"banana_ripeness_model.pkl"); (MODELS/"feature_names.json").write_text(json.dumps(FEATURE_NAMES,indent=2),encoding="utf-8")
 print(f"Saved {model_name}; held-out test accuracy: {test_metrics['Accuracy']:.3f}")
if __name__=="__main__": main()
