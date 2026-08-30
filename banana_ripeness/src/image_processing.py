from __future__ import annotations
from io import BytesIO
import cv2, numpy as np
from PIL import Image, ImageOps
from . import config
def load_image(uploaded_file):
 try:
    with Image.open(BytesIO(uploaded_file.getvalue())) as pil_image:
     pil_image=ImageOps.exif_transpose(pil_image); pil_image.verify()
    with Image.open(BytesIO(uploaded_file.getvalue())) as pil_image:
     rgb=np.array(ImageOps.exif_transpose(pil_image).convert("RGB"))
    return cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
 except (OSError,ValueError) as exc:
    raise ValueError("Invalid or corrupted image. Please upload another image.") from exc
def resize_image(image,max_dimension=config.WORKING_MAX_DIMENSION):
 h,w=image.shape[:2]; scale=min(1.,max_dimension/max(h,w)); return cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
def _component(mask):
 h,w=mask.shape; n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8); choices=[i for i in range(1,n) if stats[i,4]>=h*w*config.MIN_COMPONENT_AREA_RATIO]
 return np.where(labels==max(choices,key=lambda i:stats[i,4]),255,0).astype(np.uint8) if choices else np.zeros_like(mask)
def _grabcut(image):
 h,w=image.shape[:2]; mask=np.zeros((h,w),np.uint8); bgd=np.zeros((1,65),np.float64); fgd=np.zeros((1,65),np.float64)
 try:
  cv2.grabCut(image,mask,(max(1,w//20),max(1,h//20),max(2,w-w//10),max(2,h-h//10)),bgd,fgd,3,cv2.GC_INIT_WITH_RECT); return _component(np.where((mask==1)|(mask==3),255,0).astype(np.uint8))
 except cv2.error: return np.zeros((h,w),np.uint8)
def banana_mask(image,hsv,lab):
 h,s,v=cv2.split(hsv); H,W=h.shape; edge=max(3,min(H,W)//25); border=np.concatenate((lab[:edge].reshape(-1,3),lab[-edge:].reshape(-1,3),lab[:,:edge].reshape(-1,3),lab[:,-edge:].reshape(-1,3)))
 dist=np.linalg.norm(lab.astype(np.float32)-np.median(border,axis=0),axis=2); candidate=((((s>25)&(v>35))|((s>45)&(v<170)))&(dist>max(13,float(np.percentile(dist,42))))&~((v>242)&(s<28))).astype(np.uint8)*255
 candidate=cv2.morphologyEx(candidate,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); candidate=cv2.morphologyEx(candidate,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8)); out=_component(candidate); coverage=cv2.countNonZero(out)/out.size
 if coverage<config.MIN_FRUIT_COVERAGE or coverage>config.MAX_FRUIT_COVERAGE:
  fallback=_grabcut(image)
  if cv2.countNonZero(fallback): out=fallback
 if cv2.countNonZero(out)<out.size*config.MIN_FRUIT_COVERAGE: out=((v<245)|(s>20)).astype(np.uint8)*255
 return cv2.morphologyEx(out,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
def make_demo_image():
 image=np.full((360,520,3),(236,240,232),dtype=np.uint8); cv2.ellipse(image,(260,185),(175,66),-15,18,162,(35,210,245),-1); cv2.ellipse(image,(260,171),(164,52),-15,20,160,(60,222,250),-1)
 for x,y,r in [(230,185,5),(280,194,7),(330,160,4),(360,185,5)]: cv2.circle(image,(x,y),r,(45,78,104),-1)
 return image
def process_image(image):
 original=image.copy(); blurred=cv2.GaussianBlur(resize_image(image),config.GAUSSIAN_KERNEL,0); lab0=cv2.cvtColor(blurred,cv2.COLOR_BGR2LAB); l,a,b=cv2.split(lab0); l=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(l); working=cv2.cvtColor(cv2.merge((l,a,b)),cv2.COLOR_LAB2BGR); hsv=cv2.cvtColor(working,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(working,cv2.COLOR_BGR2LAB); mask=banana_mask(working,hsv,lab); coverage=cv2.countNonZero(mask)/float(mask.size)
 return {"original":original,"working":working,"blurred":blurred,"hsv":hsv,"lab":lab,"hsv_visual":cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB),"banana_mask":mask,"segmented":cv2.bitwise_and(working,working,mask=mask),"segmentation_coverage":round(coverage*100,1),"segmentation_uncertain":coverage<config.MIN_FRUIT_COVERAGE or coverage>config.MAX_FRUIT_COVERAGE}
def rgb(image): return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
