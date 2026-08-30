import cv2
from .image_processing import process_image
from .color_analysis import extract_features

FEATURE_NAMES=["green_percentage","yellow_percentage","brown_percentage","dark_percentage","mean_R","mean_G","mean_B","mean_H","mean_S","mean_V","mean_L","mean_a","mean_b","brown_spot_percentage","spot_count","spot_density","largest_spot_area","average_spot_area","texture_score","gradient_energy","local_variance","glcm_contrast","glcm_homogeneity","glcm_energy","glcm_correlation","fruit_area","fruit_coverage","aspect_ratio"]
def extract_features_from_image(image):
 p=process_image(image); raw=extract_features(p["working"],p["hsv"],p["lab"],p["banana_mask"])
 result={"green_percentage":raw["green"],"yellow_percentage":raw["yellow"],"brown_percentage":raw["brown"],"dark_percentage":raw["dark"]}
 result.update({name:raw[name] for name in FEATURE_NAMES if name not in result})
 return result,raw,p
def extract_features_from_path(path):
 image=cv2.imread(str(path),cv2.IMREAD_COLOR)
 if image is None: raise ValueError(f"Unreadable image: {path}")
 return extract_features_from_image(image)
