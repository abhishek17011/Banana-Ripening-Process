from pathlib import Path
import cv2
import pandas as pd

CLASS_NAMES=["green","natural","chemical"]
VALID_EXTENSIONS={".jpg",".jpeg",".png"}
def scan_dataset(dataset_dir="dataset"):
 root=Path(dataset_dir); records=[]; invalid=[]; counts={name:0 for name in CLASS_NAMES}
 for label,name in enumerate(CLASS_NAMES):
  folder=root/name
  if not folder.exists(): continue
  for path in folder.rglob("*"):
   if path.suffix.lower() not in VALID_EXTENSIONS: continue
   if cv2.imread(str(path),cv2.IMREAD_COLOR) is None: invalid.append(str(path)); continue
   records.append({"filename":str(path),"label":label,"class_name":name}); counts[name]+=1
 return records,invalid,counts
def validation_report(dataset_dir="dataset"):
 records,invalid,counts=scan_dataset(dataset_dir); table=pd.DataFrame({"Class":[x.title() for x in CLASS_NAMES],"Images":[counts[x] for x in CLASS_NAMES]})
 errors=[f"Missing or insufficient images for {name}: {counts[name]} found; at least 10 are required." for name in CLASS_NAMES if counts[name]<10]
 warnings=[]
 if any(v<50 for v in counts.values()): warnings.append("Dataset is small for reliable machine-learning training (fewer than 50 images in at least one class).")
 nonzero=[v for v in counts.values() if v]
 if nonzero and max(nonzero)/min(nonzero)>2: warnings.append("Dataset is severely imbalanced; consider collecting more images for smaller classes.")
 return records,invalid,counts,table,errors,warnings
