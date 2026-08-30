from pathlib import Path
import hashlib
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

from src.image_processing import load_image, process_image, rgb
from src.color_analysis import analyze_colors
from src.ripeness_classifier import STAGES, classify
from src.prediction import predict_ripeness
from src.banana_validator import validate_banana_image
from src.input_utils import normalize_image_input
from src import config

TRANSLATIONS = {
    "en": {
        "language": "Language", "english": "English", "marathi": "मराठी",
        "page_title": "Banana Ripeness Detection", "hero_title": "Detect Banana Ripeness Using Image Processing",
        "hero_description": "Upload a banana image and analyze its color characteristics to estimate the ripening stage.",
        "hero_eyebrow": "IMAGE PROCESSING BASED RIPENESS ESTIMATION", "take_photo": "📷 Capture Banana Image",
        "upload_heading": "Upload Banana Image", "choose_image": "Choose a JPG, JPEG, or PNG image",
        "uploaded_image": "Uploaded banana image", "camera_image": "Camera banana image",
        "demo_image": "Demo banana image", "filename": "Filename", "dimensions": "Dimensions",
        "ready": " — ready for ripeness analysis.", "analyze": "Analyze Banana", "processing": "Processing image…",
        "clear_banana": "Please capture a clear image of a banana for accurate ripeness analysis.",
        "clear_well_lit": "Please capture a clear, well-lit banana image.", "supported": "Supported image: JPG, JPEG, PNG",
        "quality_details": "📊 Quality Details", "unable_load": "Unable to load this image",
        "processing_steps": "Processing steps", "step_resize": "Read & resize image", "step_blur": "Gaussian blur",
        "step_hsv": "BGR → HSV conversion", "step_segment": "Simple banana segmentation",
        "step_color": "Green, yellow & brown analysis", "step_classify": "Ripening method classification",
        "local_processing": "All processing happens locally in this Streamlit session. This is an image-based classification.",
        "ripening_method": "RIPENING METHOD", "unavailable": "Unavailable", "no_model": "No trained three-class model is available. This result is an image-based estimate without model confidence.",
        "class": "Class", "model_confidence": "Model Confidence", "fruit_coverage": "FRUIT COVERAGE",
        "probability": "Probability (%)", "green_area": "GREEN AREA", "yellow_area": "YELLOW AREA",
        "brown_area": "BROWN AREA", "brown_spot_area": "BROWN SPOT AREA", "recommendation": "Recommendation",
        "disclaimer": "Classification is based on image/spectral features and should not be considered a definitive chemical or food-safety test.",
        "visualization": "Image Processing Visualization", "original": "1. Original Image",
        "color_mask": "3. Color Mask (green / yellow / brown)", "hsv_image": "2. HSV Processed Image",
        "brown_mask": "4. Brown Area Mask", "composition": "Banana Color Composition", "colour_class": "Colour class", "area": "Area (%)",
        "green": "Green", "yellow": "Yellow", "brown": "Brown", "dark": "Dark", "banana_detected": "🍌 Banana detected",
        "multiple_bananas": "🍌 Multiple bananas detected ({count})", "no_banana": "⚠️ No banana detected",
        "quality_low": "⚠️ Image quality is too low", "ready_analysis": "ready for ripeness analysis.",
        "stage_green": "GREEN / UNRIPE", "stage_natural": "NATURALLY RIPENED", "stage_chemical": "CHEMICALLY RIPENED",
        "desc_green": "Banana appears to be unripe based on the analyzed image features.",
        "desc_natural": "Banana is predicted as naturally ripened based on the analyzed image features.",
        "desc_chemical": "Banana is predicted as chemically ripened based on the analyzed image features.",
        "rec_green": "Allow the banana to ripen naturally before consumption.",
        "rec_natural": "Model predicts natural ripening. This is an image-based estimate.",
        "rec_chemical": "Model predicts chemical ripening. Further laboratory testing is recommended for confirmation.",
        "mode_image": "Image-based classification", "mode_trained": "Trained image-based classification",
        "reason": "Reason", "image_dimensions": "Image dimensions too small", "image_dark": "Image too dark",
        "image_blurry": "Image too blurry or low contrast", "brightness": "Brightness", "contrast": "Contrast",
            "min_required": "Minimum required", "invalid_image": "⚠️ Invalid or corrupted image. Please upload another image.",
            "uncertain": "⚠️ Classification uncertain", "uncertain_detail": "Please upload a clearer image or use additional spectral/laboratory data for confirmation."
    },
    "mr": {
        "language": "भाषा", "english": "English", "marathi": "मराठी", "page_title": "केळीच्या पिकण्याची ओळख",
        "hero_title": "प्रतिमा प्रक्रियेद्वारे केळीच्या पिकण्याची ओळख", "hero_description": "केळीचा फोटो अपलोड करा आणि पिकण्याची अवस्था जाणून घेण्यासाठी त्याच्या रंगाचे विश्लेषण करा.",
        "hero_eyebrow": "प्रतिमा प्रक्रियेवर आधारित पिकण्याचा अंदाज", "take_photo": "📷 केळीची प्रतिमा कॅमेऱ्याने घ्या", "upload_heading": "केळीची प्रतिमा अपलोड करा", "choose_image": "JPG, JPEG किंवा PNG प्रतिमा निवडा", "uploaded_image": "अपलोड केलेली केळीची प्रतिमा", "camera_image": "कॅमेऱ्याने घेतलेली केळीची प्रतिमा", "demo_image": "डेमो केळीची प्रतिमा", "filename": "फाइलचे नाव", "dimensions": "परिमाणे", "ready": " — पिकण्याच्या विश्लेषणासाठी तयार.", "analyze": "केळीच्या प्रतिमेचे विश्लेषण करा", "processing": "प्रतिमेवर प्रक्रिया सुरू आहे…", "clear_banana": "अचूक पिकण्याचे विश्लेषण करण्यासाठी कृपया केळीची स्पष्ट प्रतिमा घ्या.", "clear_well_lit": "कृपया केळीची स्पष्ट आणि पुरेशा प्रकाशातील प्रतिमा घ्या.", "supported": "समर्थित प्रतिमा: JPG, JPEG, PNG", "quality_details": "गुणवत्तेचे तपशील", "unable_load": "ही प्रतिमा उघडता आली नाही", "processing_steps": "प्रक्रिया टप्पे", "step_resize": "प्रतिमा वाचणे आणि आकार बदलणे", "step_blur": "गॉसियन ब्लर", "step_hsv": "BGR → HSV रूपांतरण", "step_segment": "केळीचे साधे विभाजन", "step_color": "हिरवा, पिवळा आणि तपकिरी रंगांचे विश्लेषण", "step_classify": "पिकण्याच्या पद्धतीचे वर्गीकरण", "local_processing": "ही सर्व प्रक्रिया Streamlit सत्रात स्थानिक पातळीवर केली जाते. हे प्रतिमेवर आधारित वर्गीकरण आहे.", "ripening_method": "पिकण्याची पद्धत", "unavailable": "उपलब्ध नाही", "no_model": "प्रशिक्षित तीन-वर्गीय मॉडेल उपलब्ध नाही. हा निकाल मॉडेलच्या विश्वास पातळीशिवाय प्रतिमेवर आधारित अंदाज आहे.", "class": "वर्ग", "model_confidence": "मॉडेलची विश्वास पातळी", "fruit_coverage": "फळाचा व्याप", "probability": "संभाव्यता (%)", "green_area": "हिरवा भाग", "yellow_area": "पिवळा भाग", "brown_area": "तपकिरी भाग", "brown_spot_area": "तपकिरी डागांचा भाग", "recommendation": "शिफारस", "disclaimer": "हे वर्गीकरण प्रतिमा आणि वर्णवैशिष्ट्यांवर आधारित आहे; हा निश्चित रासायनिक किंवा अन्न-सुरक्षा चाचणीचा पर्याय नाही.", "visualization": "प्रतिमा प्रक्रिया दृश्य", "original": "१. मूळ प्रतिमा", "color_mask": "३. रंगाचा मास्क (हिरवा / पिवळा / तपकिरी)", "hsv_image": "२. HSV प्रक्रिया केलेली प्रतिमा", "brown_mask": "४. तपकिरी भागाचा मास्क", "composition": "केळीच्या रंगांचे प्रमाण", "colour_class": "रंगाचा वर्ग", "area": "भाग (%)", "green": "हिरवा", "yellow": "पिवळा", "brown": "तपकिरी", "dark": "गडद", "banana_detected": "🍌 केळी आढळले", "multiple_bananas": "🍌 एकापेक्षा अधिक केळी आढळली ({count})", "no_banana": "⚠️ केळी आढळले नाही", "quality_low": "⚠️ प्रतिमेची गुणवत्ता खूप कमी आहे", "ready_analysis": "पिकण्याच्या विश्लेषणासाठी तयार.", "stage_green": "हिरवी / कच्ची", "stage_natural": "नैसर्गिकरीत्या पिकलेले", "stage_chemical": "रासायनिक पद्धतीने पिकवलेले", "desc_green": "विश्लेषित प्रतिमेतील वैशिष्ट्यांनुसार केळी कच्ची दिसत आहे.", "desc_natural": "प्रतिमेतील विश्लेषित वैशिष्ट्यांनुसार केळी नैसर्गिकरीत्या पिकलेली असल्याचा अंदाज आहे.", "desc_chemical": "प्रतिमेतील विश्लेषित वैशिष्ट्यांनुसार केळी रासायनिक पद्धतीने पिकवलेली असल्याचा अंदाज आहे.", "rec_green": "खाण्यापूर्वी केळी नैसर्गिकरीत्या पिकू द्या.", "rec_natural": "मॉडेलनुसार केळी नैसर्गिकरीत्या पिकलेली आहे. हा प्रतिमेवर आधारित अंदाज आहे.", "rec_chemical": "मॉडेलनुसार केळी रासायनिक पद्धतीने पिकवलेली आहे. खात्री करण्यासाठी पुढील प्रयोगशाळा तपासणीची शिफारस केली जाते.", "mode_image": "प्रतिमेवर आधारित वर्गीकरण", "mode_trained": "प्रशिक्षित प्रतिमेवर आधारित वर्गीकरण", "reason": "कारण", "image_dimensions": "प्रतिमेचे परिमाण खूप लहान आहेत", "image_dark": "प्रतिमा खूप गडद आहे", "image_blurry": "प्रतिमा अस्पष्ट आहे किंवा कॉन्ट्रास्ट कमी आहे", "brightness": "प्रकाशमानता", "contrast": "कॉन्ट्रास्ट", "min_required": "किमान आवश्यक"
    }
}

TRANSLATIONS["mr"].update({
    "invalid_image": "⚠️ अवैध किंवा खराब झालेला फोटो. कृपया दुसरा फोटो अपलोड करा.",
    "uncertain": "⚠️ वर्गीकरण अनिश्चित आहे",
    "uncertain_detail": "कृपया अधिक स्पष्ट फोटो अपलोड करा किंवा खात्रीसाठी अतिरिक्त वर्णविश्लेषण / प्रयोगशाळा तपासणीचा वापर करा."
})

# Add-on labels are kept separately so the existing translation dictionary and
# upload/validation flow remain unchanged.
TRANSLATIONS["en"].update({
    "classification": "Classification", "natural_short": "NATURAL",
    "chemical_short": "CHEMICAL", "natural_result": "NATURALLY RIPENED",
    "chemical_result": "CHEMICALLY / ARTIFICIALLY RIPENED",
    "spot_analysis": "Black/Brown Spot Analysis", "spot_coverage": "Black/Brown Spot Coverage",
    "spot_count": "Detected Spots", "spot_density": "Spot Density",
    "threshold": "Classification Threshold", "status": "Status", "above_threshold": "Above threshold",
    "below_threshold": "Below threshold", "largest_spot": "Largest Spot", "average_spot": "Average Spot Size",
    "pixels": "pixels", "segmentation": "2. Banana Segmentation", "spot_detection": "4. Black/Brown Spot Detection",
    "detected_spots": "Detected Black/Brown Spots", "color_analysis": "Color Analysis",
    "rgb_analysis": "RGB Analysis", "hsv_analysis": "HSV Analysis", "lab_analysis": "LAB Analysis",
    "texture_analysis": "Texture Analysis", "detailed_report": "Detailed Analysis Report",
    "image_information": "Image Information", "final_interpretation": "Final Interpretation",
    "banana_detected_label": "Banana detected", "yes": "Yes", "no": "No",
    "analysis_failed": "⚠️ Image analysis could not be completed. Please try a clearer banana image.",
    "not_available": "Not Available", "texture_score": "Texture score", "glcm_contrast": "GLCM contrast",
    "glcm_homogeneity": "GLCM homogeneity", "glcm_energy": "GLCM energy", "glcm_correlation": "GLCM correlation", "parameter": "Parameter", "value": "Value", "feature": "Feature",
})
TRANSLATIONS["mr"].update({
    "classification": "वर्गीकरण", "natural_short": "नैसर्गिक", "chemical_short": "रासायनिक",
    "natural_result": "नैसर्गिकरीत्या पिकलेले", "chemical_result": "रासायनिक / कृत्रिमरीत्या पिकलेले",
    "spot_analysis": "काळे/तपकिरी डागांचे विश्लेषण", "spot_coverage": "काळे/तपकिरी डागांचे प्रमाण",
    "spot_count": "आढळलेल्या डागांची संख्या", "spot_density": "डागांची घनता",
    "threshold": "वर्गीकरणाची मर्यादा", "status": "स्थिती", "above_threshold": "मर्यादेपेक्षा जास्त",
    "below_threshold": "मर्यादेपेक्षा कमी", "largest_spot": "सर्वात मोठा डाग", "average_spot": "डागाचा सरासरी आकार",
    "pixels": "पिक्सेल", "segmentation": "२. केळीचे विभाजन", "spot_detection": "४. काळे/तपकिरी डाग शोध",
    "detected_spots": "आढळलेले काळे/तपकिरी डाग", "color_analysis": "रंग विश्लेषण",
    "rgb_analysis": "RGB विश्लेषण", "hsv_analysis": "HSV विश्लेषण", "lab_analysis": "LAB विश्लेषण",
    "texture_analysis": "पोत विश्लेषण", "detailed_report": "सविस्तर विश्लेषण अहवाल",
    "image_information": "प्रतिमा माहिती", "final_interpretation": "अंतिम निष्कर्ष",
    "banana_detected_label": "केळी आढळले", "yes": "होय", "no": "नाही",
    "analysis_failed": "⚠️ प्रतिमेचे विश्लेषण पूर्ण होऊ शकले नाही. कृपया केळीचा अधिक स्पष्ट फोटो वापरून पहा.",
    "not_available": "उपलब्ध नाही", "texture_score": "पोत गुणांक", "glcm_contrast": "GLCM कॉन्ट्रास्ट",
    "glcm_homogeneity": "GLCM समरूपता", "glcm_energy": "GLCM ऊर्जा", "glcm_correlation": "GLCM सहसंबंध", "parameter": "घटक", "value": "मूल्य", "feature": "वैशिष्ट्य",
})
TRANSLATIONS["en"].update({
    "segmentation_uncertain": "⚠️ Banana segmentation is uncertain. Please upload a clearer image."
})
TRANSLATIONS["mr"].update({
    "segmentation_uncertain": "⚠️ केळीचे विभाजन अनिश्चित आहे. कृपया अधिक स्पष्ट फोटो अपलोड करा."
})

if "language" not in st.session_state:
    st.session_state.language = "en"

def t(key, **values):
    return TRANSLATIONS[st.session_state.language][key].format(**values)

def format_number(value):
    """Format existing numeric results consistently without altering their values."""
    return f"{float(value):.2f}"

def format_count(value):
    return str(int(value))

def display_table(data):
    frame = pd.DataFrame(data)
    value_column = frame.columns[-1]
    return st.dataframe(
        frame.style.set_properties(subset=[value_column], **{"text-align": "right"}),
        hide_index=True,
        use_container_width=True,
    )

def localized_result(result):
    if st.session_state.language == "en":
        return result
    stage_keys = {STAGES[0]: "green", STAGES[1]: "natural", STAGES[2]: "chemical"}
    key = stage_keys.get(result.get("stage"))
    if key is None:
        return result
    return {**result, "stage": t(f"stage_{key}"), "description": t(f"desc_{key}"), "recommendation": t(f"rec_{key}"), "mode": t("mode_image") if result.get("mode") == "Image-based classification" else t("mode_trained")}

def localized_validation_message(message):
    if message.startswith("🍌 Multiple bananas detected"):
        return t("multiple_bananas", count=message.split("(")[-1].rstrip(")"))
    message_keys = {"🍌 Banana detected": "banana_detected", "⚠️ No banana detected": "no_banana", "⚠️ Image quality is too low": "quality_low"}
    return t(message_keys[message]) if message in message_keys else message

st.set_page_config(page_title=t("page_title"), page_icon="", layout="wide")
st.markdown("""<style>
 .stApp { background: #f8f6ed; color: #12271b; }
 h1 { color:#10291b; font-weight:800; letter-spacing:-.04em; font-size:2.35rem; }
 h2, h3 { color:#163a27; font-weight:750; letter-spacing:-.015em; line-height:1.25; }
 .hero { background:linear-gradient(110deg,#fdfcf6,#fff4bd); border-radius:28px; padding:2.6rem 3rem; margin-bottom:2rem; }
 .card { background:#fffefa; padding:1.35rem; border:1px solid #e0e3d8; border-radius:20px; box-shadow:0 6px 20px #35512b12; }
 .metric-card { background:#eef5e8; padding:1rem; border-radius:16px; border:1px solid #dbe7d5; text-align:center; min-height:104px; }
 .stage { display:flex; gap:6px; margin:16px 0 4px; } .stage span { flex:1; text-align:center; padding:9px 3px; font-size:11px; font-weight:700; border-radius:12px; background:#e2e8d9; color:#66766b; }
 .stage .active { background:#f4c914; color:#18281c; box-shadow:0 2px 8px #d5ad1640; }
 .disclaimer { color:#65766a; font-size:.85rem; line-height:1.5; } div.stButton > button { background:#168344; color:#fff; border:0; border-radius:12px; font-weight:700; padding:.55rem 1.1rem; } div.stButton > button:hover { background:#0d6933; color:#fff; }
 [data-testid="stFileUploader"] { border:1px solid #dbe7d5; border-radius:14px; padding:.35rem .65rem; background:#fffefa; }
 [data-testid="stDataFrame"] { border:1px solid #e0e3d8; border-radius:12px; overflow:hidden; font-size:.94rem; }
 [data-testid="stDataFrame"] [role="columnheader"] { font-weight:750; background:#f4f7f0; }
 [data-testid="stDataFrame"] [role="gridcell"] { line-height:1.45; }
</style>""", unsafe_allow_html=True)

def stage_indicator(active: int):
    stage_keys = ["green", "natural", "chemical"]
    cells = "".join(f'<span class="{"active" if i == active else ""}">{t(f"stage_{stage_keys[i - 1]}")}</span>' for i, name in enumerate(STAGES, 1))
    st.markdown(f'<div class="stage">{cells}</div>', unsafe_allow_html=True)

def run_analysis(image, filename):
    processed = process_image(image)
    colors = analyze_colors(processed["hsv"], processed["banana_mask"], processed["working"], processed["lab"])
    spot_coverage = float(colors.get("brown_spot_percentage", 0.0))
    is_natural = spot_coverage >= config.NATURAL_SPOT_THRESHOLD
    result = {"classification": "natural" if is_natural else "chemical", "spot_coverage": spot_coverage}
    st.session_state.analysis = {**processed, **colors, "result": result, "filename": filename, "dimensions": (image.shape[1], image.shape[0])}


def render_image_input(image, filename, source_label):
    if image is None:
        return

    source_key = {"Uploaded": "uploaded_image", "Camera": "camera_image", "Demo": "demo_image"}[source_label]
    st.image(rgb(image), caption=t(source_key), use_container_width=True)
    st.caption(f"**{t('filename')}:** {filename}    •    **{t('dimensions')}:** {format_count(image.shape[1])} × {format_count(image.shape[0])} px")

    validation_result = validate_banana_image(image)
    st.session_state.validation = validation_result

    if validation_result["quality_ok"]:
        if validation_result["is_banana"]:
            st.success(localized_validation_message(validation_result["message"]) + t("ready"))

            if st.button(t("analyze"), type="primary", use_container_width=True):
                with st.spinner(t("processing")):
                    try:
                        run_analysis(image, filename)
                    except (cv2.error, ValueError, ArithmeticError, TypeError):
                        st.session_state.pop("analysis", None)
                        st.error(t("analysis_failed"))
        else:
            st.error(localized_validation_message(validation_result["message"]))
            st.info(f"{t('clear_banana')}\n\n{t('supported')}")
    else:
        st.error(localized_validation_message(validation_result["message"]))
        st.info(f"{t('clear_well_lit')}\n\n{t('supported')}")

        if validation_result.get("quality_issues"):
            with st.expander(t("quality_details")):
                for key, value in validation_result["quality_issues"].items():
                    if st.session_state.language == "en":
                        st.text(f"{key}: {value}")
                        continue
                    issue_keys = {"reason": "reason", "width": "dimensions", "height": "dimensions", "min_required": "min_required", "brightness": "brightness", "contrast": "contrast"}
                    reason_keys = {"Image dimensions too small": "image_dimensions", "Image too dark": "image_dark", "Image too blurry or low contrast": "image_blurry"}
                    display_key = t(issue_keys.get(key, key))
                    display_value = t(reason_keys[value]) if key == "reason" and value in reason_keys else value
                    st.text(f"{display_key}: {display_value}")

st.selectbox(t("language"), [t("english"), t("marathi")], key="language_choice", index=0 if st.session_state.language == "en" else 1, on_change=lambda: setattr(st.session_state, "language", "mr" if st.session_state.language_choice == "मराठी" else "en"))
if st.session_state.language == "mr":
    st.markdown("<style>html, body, [class*='st-'] { font-family: 'Noto Sans Devanagari', sans-serif; }</style>", unsafe_allow_html=True)

st.markdown(f"""<div class="hero"><h1> {t("hero_title")}</h1>
<p style="font-size:1.15rem;color:#65766a;max-width:720px">{t("hero_description")}</p>
<p style="color:#168344;font-weight:700;letter-spacing:.08em;font-size:.76rem">{t("hero_eyebrow")}</p></div>""", unsafe_allow_html=True)

left, right = st.columns([1.15, .85], gap="large")
with left:
    camera = st.camera_input(t("take_photo"))
    st.subheader(t("upload_heading"))
    uploaded = st.file_uploader(t("choose_image"), type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if uploaded:
        input_token = ("upload", hashlib.sha256(uploaded.getvalue()).hexdigest())
    elif camera is not None:
        input_token = ("camera", hashlib.sha256(camera.getvalue()).hexdigest())
    else:
        input_token = None
    if input_token != st.session_state.get("input_token"):
        st.session_state.pop("analysis", None)
        st.session_state.pop("validation", None)
        st.session_state.input_token = input_token
    image = None
    filename = ""
    try:
        if uploaded:
            image, filename = load_image(uploaded), uploaded.name
            render_image_input(image, filename, "Uploaded")
        elif camera is not None:
            image = cv2.cvtColor(normalize_image_input(camera), cv2.COLOR_RGB2BGR)
            filename = "camera_capture.png"
            render_image_input(image, filename, "Camera")
    except (ValueError, OSError) as exc:
        if "Invalid or corrupted image" in str(exc):
            st.error(t("invalid_image"))
        else:
            st.error(f"{t('unable_load')}: {exc}")
with right:
    st.markdown(f'<div class="card"><h3>{t("processing_steps")}</h3><ol><li>{t("step_resize")}</li><li>{t("step_blur")}</li><li>{t("step_hsv")}</li><li>{t("step_segment")}</li><li>{t("step_color")}</li><li>{t("step_classify")}</li></ol><p class="disclaimer">{t("local_processing")}</p></div>', unsafe_allow_html=True)

analysis = st.session_state.get("analysis")
if analysis and False:  # Legacy result presentation retained but superseded by the spot-rule report below.
    natural = analysis["result"]["classification"] == "natural"
    spot_coverage = float(analysis["brown_spot_percentage"])
    threshold = float(config.NATURAL_SPOT_THRESHOLD)
    result_title = t("natural_result" if natural else "chemical_result")
    result_short = t("natural_short" if natural else "chemical_short")
    st.divider()
    st.markdown(f"## {t('ripening_method')}")
    confidence = f'{r["confidence"] * 100:.1f}%' if r["confidence"] is not None and analysis["result"].get("mode", "").startswith("Trained") else t("unavailable")
    if analysis["result"].get("mode") == "Image-based classification":
        st.warning(t("no_model"))
    a, b = st.columns([3, 1])
    with a:
        st.markdown(f'<div class="card"><h2>🟨 {r["stage"]}</h2><p>{t("class")} {r["stage_number"]} / 3</p><p>{r["description"]}</p><p class="disclaimer">{r["mode"]}</p></div>', unsafe_allow_html=True)
        if analysis.get("segmentation_uncertain"):
            st.warning(t("segmentation_uncertain"))
        elif analysis["result"].get("uncertain"):
            st.warning(f"{t('uncertain')} . {t('uncertain_detail')}")
    with b:
        st.metric(t("fruit_coverage"), f'{analysis["fruit_coverage"]}%')
    stage_indicator(r["stage_number"])
    if r.get("probabilities"):
        stage_key = {STAGES[0]: "green", STAGES[1]: "natural", STAGES[2]: "chemical"}
        probability_classes = [t(f"stage_{stage_key[name]}") for name in r["probabilities"]]
        st.dataframe(pd.DataFrame({t("class"): probability_classes, t("probability"): [round(value * 100, 1) for value in r["probabilities"].values()]}), hide_index=True, use_container_width=True)
    metrics = st.columns(4)
    for col, label, value in zip(metrics, [t("green_area"), t("yellow_area"), t("brown_area"), t("brown_spot_area")], [analysis["green"], analysis["yellow"], analysis["brown"], analysis["brown_area"]]):
        with col:
            st.markdown(f'<div class="metric-card"><small>{label}</small><h3>{value}%</h3></div>', unsafe_allow_html=True)
    st.markdown(f"### {t('recommendation')}")
    st.success(r["recommendation"])
    st.markdown(f'<p class="disclaimer">{t("disclaimer")}</p>', unsafe_allow_html=True)
    st.markdown(f"### {t('visualization')}")
    v1, v2 = st.columns(2)
    with v1:
        st.image(rgb(analysis["original"]), caption=t("original"), use_container_width=True)
        st.image(analysis["color_mask"], channels="BGR", caption=t("color_mask"), use_container_width=True)
    with v2:
        st.image(analysis["hsv_visual"], caption=t("hsv_image"), use_container_width=True)
        st.image(analysis["brown_mask"], caption=t("brown_mask"), use_container_width=True)
    st.markdown(f"### {t('composition')}")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    values = [analysis["green"], analysis["yellow"], analysis["brown"]]
    ax.pie(values, labels=[t("green"), t("yellow"), t("brown")], colors=["#4fa65a", "#f5cb21", "#905631"], autopct="%1.1f%%", startangle=90, wedgeprops={"width": .46, "edgecolor": "white"})
    ax.set(aspect="equal")
    st.pyplot(fig, use_container_width=False)
    st.dataframe(pd.DataFrame({t("colour_class"): [t("green"), t("yellow"), t("brown"), t("dark")], t("area"): [analysis["green"], analysis["yellow"], analysis["brown"], analysis["dark"]]}), hide_index=True, use_container_width=True)

if analysis:
    natural = analysis["result"]["classification"] == "natural"
    spot_coverage = float(analysis["brown_spot_percentage"])
    threshold = float(config.NATURAL_SPOT_THRESHOLD)
    result_title = t("natural_result" if natural else "chemical_result")
    result_short = t("natural_short" if natural else "chemical_short")
    status = t("above_threshold" if natural else "below_threshold")
    icon = "🍌" if natural else "🧪"

    st.divider()
    st.markdown(f"## {t('ripening_method')}")
    st.markdown(f'<div class="card"><h2>{icon} {result_title}</h2><p><b>{t("classification")}:</b> {result_short}</p><p class="disclaimer">{t("local_processing")}</p></div>', unsafe_allow_html=True)
    if analysis.get("segmentation_uncertain"):
        st.warning(t("segmentation_uncertain"))
    summary = st.columns(4)
    for col, label, value in zip(summary, [t("spot_coverage"), t("spot_count"), t("threshold"), t("status")], [f"{format_number(spot_coverage)}%", format_count(analysis["spot_count"]), f"{format_number(threshold)}%", status]):
        with col:
            st.markdown(f'<div class="metric-card"><small>{label}</small><h3>{value}</h3></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="disclaimer">{t("disclaimer")}</p>', unsafe_allow_html=True)

    st.markdown(f"### {t('visualization')}")
    v1, v2 = st.columns(2)
    with v1:
        st.image(rgb(analysis["original"]), caption=t("original"), use_container_width=True)
        st.image(analysis["hsv_visual"], caption=t("hsv_image"), use_container_width=True)
    with v2:
        st.image(analysis["segmented"], channels="BGR", caption=t("segmentation"), use_container_width=True)
        st.image(analysis["brown_spot_mask"], caption=f'{t("detected_spots")} — {t("spot_coverage")}: {format_number(spot_coverage)}%', use_container_width=True)

    st.markdown(f"### {t('composition')}")
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    values = [analysis["green"], analysis["yellow"], analysis["brown"], analysis["dark"]]
    labels = [t("green"), t("yellow"), t("brown"), t("dark")]
    wedges, _, _ = ax.pie(values, colors=["#5B9A68", "#E4B843", "#9A5B3B", "#30343B"], autopct="%1.2f%%", pctdistance=.76, startangle=90, wedgeprops={"width": .46, "edgecolor": "white", "linewidth": 1.25}, textprops={"fontsize": 10, "fontweight": "medium"})
    ax.set(aspect="equal")
    ax.legend(wedges, labels, loc="lower center", bbox_to_anchor=(.5, -.15), ncol=2, frameon=False, fontsize=10)
    fig.subplots_adjust(bottom=.2)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    display_table({t("colour_class"): labels, t("area"): [f"{format_number(value)}%" for value in values]})

    st.markdown(f"### {t('spot_analysis')}")
    display_table({t("parameter"): [t("spot_coverage"), t("spot_count"), t("spot_density"), t("largest_spot"), t("average_spot")], t("value"): [f"{format_number(spot_coverage)}%", format_count(analysis["spot_count"]), format_number(analysis["spot_density"]), f'{format_number(analysis["largest_spot_area"])} {t("pixels")}', f'{format_number(analysis["average_spot_area"])} {t("pixels")}']})
    feature_columns = st.columns(3)
    feature_sets = [(t("rgb_analysis"), [("R", analysis["mean_R"]), ("G", analysis["mean_G"]), ("B", analysis["mean_B"])]), (t("hsv_analysis"), [("H", analysis["mean_H"]), ("S", analysis["mean_S"]), ("V", analysis["mean_V"])]), (t("lab_analysis"), [("L", analysis["mean_L"]), ("A", analysis["mean_a"]), ("B", analysis["mean_b"])])]
    for column, (heading, rows) in zip(feature_columns, feature_sets):
        with column:
            st.markdown(f"### {heading}")
            display_table({t("feature"): [label for label, _ in rows], t("value"): [format_number(value) for _, value in rows]})
    st.markdown(f"### {t('texture_analysis')}")
    texture_values = [analysis["texture_score"], analysis["glcm_contrast"], analysis["glcm_homogeneity"], analysis["glcm_energy"], analysis["glcm_correlation"]]
    display_table({t("feature"): [t("texture_score"), t("glcm_contrast"), t("glcm_homogeneity"), t("glcm_energy"), t("glcm_correlation")], t("value"): [format_number(value) for value in texture_values]})

    interpretation = (f"Black/brown spots were detected on the banana surface. The detected spot coverage is {spot_coverage:.2f}%, which is above the configured threshold of {threshold:.2f}%. Based on this image-processing rule, the banana is classified as naturally ripened." if natural else f"Black/brown spots were detected at only {spot_coverage:.2f}%, which is below the configured threshold of {threshold:.2f}%. Based on this image-processing rule, the banana is classified as chemically/artificially ripened.")
    if st.session_state.language == "mr":
        interpretation = (f"केळीच्या पृष्ठभागावर काळे/तपकिरी डाग आढळले. डागांचे प्रमाण {spot_coverage:.2f}% आहे, जे {threshold:.2f}% या निश्चित मर्यादेपेक्षा जास्त आहे. या प्रतिमा-प्रक्रिया नियमावरून केळी नैसर्गिकरीत्या पिकलेली म्हणून वर्गीकृत केली आहे." if natural else f"काळे/तपकिरी डागांचे प्रमाण फक्त {spot_coverage:.2f}% आहे, जे {threshold:.2f}% या निश्चित मर्यादेपेक्षा कमी आहे. या प्रतिमा-प्रक्रिया नियमावरून केळी रासायनिक/कृत्रिमरीत्या पिकवलेली म्हणून वर्गीकृत केली आहे.")
    st.markdown(f"### {t('final_interpretation')}")
    st.info(interpretation)

    st.divider()
    st.markdown(f"## {t('detailed_report')}")
    st.markdown(f"### {t('image_information')}")
    display_table({t("parameter"): [t("filename"), t("dimensions"), t("banana_detected_label")], t("value"): [analysis["filename"], f'{format_count(analysis["dimensions"][0])} × {format_count(analysis["dimensions"][1])} px', t("yes")]})
    st.markdown(f"### {t('classification')}")
    display_table({t("parameter"): [t("classification"), t("spot_coverage"), t("threshold"), t("spot_count")], t("value"): [result_title, f"{format_number(spot_coverage)}%", f"{format_number(threshold)}%", format_count(analysis["spot_count"])]})
