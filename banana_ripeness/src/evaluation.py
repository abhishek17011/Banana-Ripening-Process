"""Evaluate the existing image-processing ripening-method rule on labeled images."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import cv2
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from . import config
from .color_analysis import analyze_colors
from .image_processing import process_image

CLASS_NAMES = ("Natural", "Chemical / Artificial")
CLASS_FOLDERS = {
    "Natural": ("natural",),
    "Chemical / Artificial": ("chemical", "artificial", "chemical_artificial"),
}
PSEUDO_CLASS_FOLDERS = {
    "Natural": ("ripe", "overripe"),
    "Chemical / Artificial": ("spoiled",),
}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _find_folder(root: Path, names: Tuple[str, ...]) -> Optional[Path]:
    for name in names:
        folder = root / name
        if folder.is_dir():
            return folder
    return None


def _iter_images(folder: Path):
    yield from sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def evaluate_dataset(
    dataset_dir: Union[str, Path],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Process one image at a time and calculate metrics from real labels when available."""
    root = Path(dataset_dir)
    labeled_files: List[Tuple[Path, str]] = []
    missing_classes: List[str] = []
    ground_truth_valid = False
    pseudo_labeled = False

    for class_name, folder_names in CLASS_FOLDERS.items():
        folder = _find_folder(root, folder_names)
        if folder is None:
            missing_classes.append(class_name)
            continue
        ground_truth_valid = True
        labeled_files.extend((path, class_name) for path in _iter_images(folder))

    if not labeled_files:
        pseudo_labeled = True
        for class_name, folder_names in PSEUDO_CLASS_FOLDERS.items():
            folder = _find_folder(root, folder_names)
            if folder is None:
                continue
            labeled_files.extend((path, class_name) for path in _iter_images(folder))

    if labeled_files and not ground_truth_valid:
        pseudo_labeled = True

    results: List[dict] = []
    skipped: List[dict] = []
    total = len(labeled_files)
    for index, (path, actual) in enumerate(labeled_files, start=1):
        try:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("unsupported or corrupted image")
            processed = process_image(image)
            colors = analyze_colors(
                processed["hsv"],
                processed["banana_mask"],
                processed["working"],
                processed["lab"],
            )
            spot_percentage = float(colors.get("brown_spot_percentage", 0.0))
            predicted = CLASS_NAMES[0] if spot_percentage >= config.NATURAL_SPOT_THRESHOLD else CLASS_NAMES[1]
            results.append({
                "filename": str(path),
                "actual": actual,
                "predicted": predicted,
                "spot_percentage": spot_percentage,
            })
        except (cv2.error, OSError, ValueError, ArithmeticError, TypeError) as exc:
            skipped.append({"filename": str(path), "reason": str(exc)})
        finally:
            if progress_callback:
                progress_callback(index, total)

    if not results:
        return {
            "available": False,
            "missing_classes": missing_classes,
            "results": [],
            "skipped": skipped,
            "ground_truth_valid": ground_truth_valid,
            "pseudo_labeled": pseudo_labeled,
            "metrics_valid": ground_truth_valid,
        }

    actual_values = [row["actual"] for row in results]
    predicted_values = [row["predicted"] for row in results]
    matrix = confusion_matrix(actual_values, predicted_values, labels=list(CLASS_NAMES))
    correct = int(sum(actual == predicted for actual, predicted in zip(actual_values, predicted_values)))
    total_predictions = len(results)
    accuracy = correct / total_predictions if total_predictions else 0.0
    precision = recall = f1 = 0.0
    if ground_truth_valid:
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_values,
            predicted_values,
            labels=list(CLASS_NAMES),
            average="weighted",
            zero_division=0,
        )

    return {
        "available": True,
        "missing_classes": missing_classes,
        "results": results,
        "skipped": skipped,
        "confusion_matrix": matrix,
        "total_images": total_predictions,
        "correct_predictions": correct,
        "incorrect_predictions": total_predictions - correct,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "table": pd.DataFrame(results),
        "ground_truth_valid": ground_truth_valid,
        "pseudo_labeled": pseudo_labeled,
        "metrics_valid": ground_truth_valid,
    }