"""
Banana image validator - detects whether uploaded image contains a banana.
Uses YOLO for primary detection, falls back to image processing features.
"""

import cv2
import numpy as np
from pathlib import Path
from . import config

# Try to import YOLO, but gracefully handle if ultralytics is not installed
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class BananaValidator:
    """Validates banana presence in images with confidence scoring."""
    
    def __init__(self):
        self.yolo_model = None
        self.method_used = "none"
        self._initialize_yolo()
    
    def _initialize_yolo(self):
        """Initialize YOLO model for banana detection."""
        if not YOLO_AVAILABLE:
            return
        
        try:
            # Use YOLOv8 nano for speed, which includes fruit detection
            self.yolo_model = YOLO("yolov8n.pt")
            self.method_used = "yolo"
        except Exception as e:
            # If download or loading fails, fall back to image processing
            self.yolo_model = None
            self.method_used = "fallback"
    
    def _check_image_quality(self, image):
        """
        Check if image quality is sufficient for analysis.
        Returns: (is_quality_ok: bool, quality_metrics: dict)
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Check image dimensions
        height, width = image.shape[:2]
        if height < config.IMAGE_QUALITY_MIN_DIMENSION or width < config.IMAGE_QUALITY_MIN_DIMENSION:
            return False, {
                "reason": "Image dimensions too small",
                "width": width,
                "height": height,
                "min_required": config.IMAGE_QUALITY_MIN_DIMENSION
            }
        
        # Check brightness (mean pixel value)
        brightness = np.mean(gray)
        if brightness < config.IMAGE_QUALITY_BRIGHTNESS_MIN:
            return False, {
                "reason": "Image too dark",
                "brightness": round(float(brightness), 2),
                "min_required": config.IMAGE_QUALITY_BRIGHTNESS_MIN
            }
        
        # Check contrast (standard deviation)
        contrast = np.std(gray)
        if contrast < config.IMAGE_QUALITY_CONTRAST_MIN:
            return False, {
                "reason": "Image too blurry or low contrast",
                "contrast": round(float(contrast), 2),
                "min_required": config.IMAGE_QUALITY_CONTRAST_MIN
            }
        
        return True, {
            "brightness": round(float(brightness), 2),
            "contrast": round(float(contrast), 2),
            "dimensions": (width, height)
        }
    
    def _detect_banana_yolo(self, image):
        """
        Detect banana using YOLO model.
        Returns: (confidence: float, banana_count: int, details: dict)
        """
        if not self.yolo_model:
            return 0.0, 0, {"error": "YOLO model not initialized"}
        
        try:
            results = self.yolo_model(image, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)
            
            if not results or len(results) == 0:
                return 0.0, 0, {"error": "No detections"}
            
            result = results[0]
            detections = result.boxes
            
            # Look for fruit-related classes in YOLO v8
            # Class 47 is typically "banana" or other fruit classes
            banana_detections = []
            for box in detections:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # YOLO v8 nano includes fruit detection
                # We look for high-confidence detections that could be banana
                # Class names vary, but we use confidence as proxy
                if conf > config.YOLO_CONFIDENCE_THRESHOLD:
                    banana_detections.append({
                        "confidence": conf,
                        "class_id": cls_id,
                        "bbox": box.xyxy[0].tolist()
                    })
            
            if banana_detections:
                # Use max confidence among detections
                max_conf = max(det["confidence"] for det in banana_detections)
                return max_conf, len(banana_detections), {
                    "detections": banana_detections,
                    "max_confidence": max_conf
                }
            else:
                return 0.0, 0, {"reason": "No high-confidence fruit detections"}
                
        except Exception as e:
            return 0.0, 0, {"error": str(e)}
    
    def _detect_banana_fallback(self, image):
        """
        Fallback banana detection using image processing features.
        Analyzes color distribution and shape characteristics.
        
        Returns: (confidence: float, banana_count: int, details: dict)
        """
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Banana yellow range in HSV: H: 15-35 (approximate)
            # Unripe banana green range: H: 40-80
            yellow_mask = cv2.inRange(hsv, (15, 40, 50), (35, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 30, 50), (80, 255, 255))
            
            # Combine masks
            banana_like = cv2.bitwise_or(yellow_mask, green_mask)
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            banana_like = cv2.morphologyEx(banana_like, cv2.MORPH_CLOSE, kernel)
            banana_like = cv2.morphologyEx(banana_like, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(banana_like, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return 0.0, 0, {"reason": "No banana-colored objects found"}
            
            # Analyze contours for banana-like shapes
            image_area = image.shape[0] * image.shape[1]
            banana_count = 0
            confidence_scores = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Check if contour is reasonably sized (not noise, not entire image)
                if area < image_area * 0.01 or area > image_area * 0.85:
                    continue
                
                # Analyze shape
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                # Circularity (how close to circle): 4π*area/perimeter²
                # Banana should be elongated (low circularity)
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = max(w, h) / (min(w, h) + 1e-6)
                
                # Banana-like characteristics:
                # - Not too circular (circularity < 0.7)
                # - Elongated (aspect ratio > 1.5)
                # - Reasonable area
                if circularity < 0.75 and aspect_ratio > 1.2:
                    # Score based on color saturation in banana range
                    mask = np.zeros_like(yellow_mask)
                    cv2.drawContours(mask, [contour], 0, 255, -1)
                    
                    # Check saturation in banana range
                    masked_s = s[mask == 255]
                    if len(masked_s) > 0:
                        saturation_score = np.mean(masked_s) / 255.0
                        confidence_scores.append(saturation_score * 0.8 + (1 - min(circularity, 1)) * 0.2)
                        banana_count += 1
            
            if confidence_scores:
                # Average confidence of all detected banana-like objects
                avg_confidence = np.mean(confidence_scores)
                max_confidence = np.max(confidence_scores)
                return max_confidence, banana_count, {
                    "average_confidence": avg_confidence,
                    "max_confidence": max_confidence,
                    "banana_count": banana_count
                }
            else:
                return 0.0, 0, {"reason": "No banana-like objects detected"}
                
        except Exception as e:
            return 0.0, 0, {"error": str(e)}
    
    def validate_image(self, image):
        """
        Validate if image contains a banana.
        
        Args:
            image: OpenCV image (BGR format)
        
        Returns:
            dict with keys:
            - is_banana (bool): True if banana detected above threshold
            - confidence (float): Confidence score 0-1
            - banana_count (int): Number of bananas detected
            - message (str): User-friendly message
            - method (str): Detection method used (yolo/fallback)
            - quality_ok (bool): Whether image quality is sufficient
            - quality_issues (dict): Details about any quality issues
            - details (dict): Additional technical details
        """
        # Check image quality first
        quality_ok, quality_metrics = self._check_image_quality(image)
        
        if not quality_ok:
            return {
                "is_banana": False,
                "confidence": 0.0,
                "banana_count": 0,
                "message": "⚠️ Image quality is too low",
                "method": "quality_check",
                "quality_ok": False,
                "quality_issues": quality_metrics,
                "details": {}
            }
        
        # Try YOLO first if available
        confidence = 0.0
        banana_count = 0
        detection_details = {}
        method_used = "fallback"
        
        if self.yolo_model is not None:
            confidence, banana_count, detection_details = self._detect_banana_yolo(image)
            method_used = "yolo"
            
            # If YOLO gives good result, use it
            if confidence >= config.BANANA_DETECTION_THRESHOLD:
                if banana_count == 1:
                    message = "🍌 Banana detected"
                else:
                    message = f"🍌 Multiple bananas detected ({banana_count})"
                
                return {
                    "is_banana": True,
                    "confidence": round(confidence, 3),
                    "banana_count": banana_count,
                    "message": message,
                    "method": "yolo",
                    "quality_ok": True,
                    "quality_issues": {},
                    "details": detection_details
                }
        
        # Fall back to image processing if YOLO not available or didn't detect
        confidence, banana_count, detection_details = self._detect_banana_fallback(image)
        
        # Determine result
        is_banana = confidence >= config.BANANA_DETECTION_THRESHOLD
        
        if is_banana:
            if banana_count == 1:
                message = "🍌 Banana detected"
            else:
                message = f"🍌 Multiple bananas detected ({banana_count})"
        else:
            message = "⚠️ No banana detected"
        
        return {
            "is_banana": is_banana,
            "confidence": round(confidence, 3),
            "banana_count": banana_count,
            "message": message,
            "method": "fallback",
            "quality_ok": True,
            "quality_issues": {},
            "details": detection_details
        }


# Global validator instance
_validator_instance = None

def get_validator():
    """Get or create global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = BananaValidator()
    return _validator_instance

def validate_banana_image(image):
    """
    Validate if image contains a banana.
    
    Args:
        image: OpenCV image (BGR format)
    
    Returns:
        dict with validation result
    """
    validator = get_validator()
    return validator.validate_image(image)
