from pathlib import Path

from src.evaluation import evaluate_dataset


def test_existing_project_dataset_is_marked_pseudo_labeled():
    dataset_root = Path(__file__).resolve().parents[1] / "dataset"
    natural_dir = dataset_root / "ripe"
    chemical_dir = dataset_root / "spoiled"

    assert natural_dir.exists(), "Expected the existing dataset to be present"
    assert chemical_dir.exists(), "Expected the existing dataset to be present"

    result = evaluate_dataset(dataset_root)

    assert result["available"] is True
    assert result["ground_truth_valid"] is False
    assert result["pseudo_labeled"] is True
    assert result["confusion_matrix"].size > 0
    assert result["metrics_valid"] is False
