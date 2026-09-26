import csv
from pathlib import Path

from src.evaluation import evaluate_dataset


def test_ripeness_categories_are_not_inferred_as_ripening_method_labels():
    dataset_root = Path(__file__).resolve().parents[1] / "dataset"
    result = evaluate_dataset(dataset_root)

    assert result["available"] is False
    assert result["ground_truth_valid"] is False
    assert result["pseudo_labeled"] is False
    assert result["metrics_valid"] is False


def test_pseudo_labeled_manifest_does_not_report_performance_metrics(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    image_path = next((project_root / "dataset" / "ripe").glob("*.jpg"))
    evaluation_root = tmp_path / "evaluation_dataset_pseudo_labeled"
    evaluation_root.mkdir()
    manifest_path = evaluation_root / "manifest.csv"

    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("original_image_path", "source_category", "evaluation_label", "image_filename", "label_type"),
        )
        writer.writeheader()
        writer.writerow({
            "original_image_path": str(image_path),
            "source_category": "ripe",
            "evaluation_label": "Natural",
            "image_filename": image_path.name,
            "label_type": "PSEUDO-LABEL",
        })

    result = evaluate_dataset(evaluation_root)

    assert result["available"] is True
    assert result["ground_truth_valid"] is False
    assert result["pseudo_labeled"] is True
    assert result["confusion_matrix"].sum() == 1
    assert result["accuracy"] is None
    assert result["precision"] is None
    assert result["recall"] is None
    assert result["f1"] is None
    assert result["metrics_valid"] is False
