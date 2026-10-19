"""Committed artifacts, README and documented commands must agree with the code."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from src.predict import validate_metadata
from tests.synthetic import FEATURES

ROOT = Path(__file__).resolve().parent.parent


def test_committed_metadata_is_valid_and_matches_schema():
    meta = validate_metadata(json.loads((ROOT / "models" / "metadata.json").read_text()))
    assert meta["feature_names"] == FEATURES               # Time, V1..V28, Amount, in that order
    assert 0.0 < meta["threshold"] < 1.0


def test_committed_metadata_and_metrics_report_the_same_threshold():
    meta = json.loads((ROOT / "models" / "metadata.json").read_text())
    metrics = json.loads((ROOT / "results" / "final_test_metrics.json").read_text())
    assert meta["threshold"] == metrics["threshold"]
    assert meta["threshold_selection_rule"] == metrics["threshold_selection_rule"]
    cm = metrics["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == metrics["test_fraud_cases"]
    assert sum(cm.values()) == metrics["test_rows"]


def test_stale_test_tuned_plots_are_archived_with_an_explanation():
    """The old 0.003-threshold figure is kept only under assets/legacy, with a note saying why."""
    legacy = ROOT / "assets" / "legacy"
    assert (legacy / "threshold_tradeoff.png").exists()
    note = (legacy / "README.md").read_text()
    assert "0.003" in note and "test set" in note


def test_readme_relative_links_and_images_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)\s]+)\)", readme)
    local = [t for t in targets if not t.startswith(("http", "#", "mailto"))]
    assert local, "expected at least one local link"
    missing = [t for t in local if not (ROOT / t.split("#")[0]).exists()]
    assert not missing, f"README links to missing files: {missing}"


def test_readme_mentions_only_existing_source_modules():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for module in set(re.findall(r"src/(\w+)\.py", readme)):
        assert (ROOT / "src" / f"{module}.py").exists(), module
    for path in ("app/app.py", "data/README.md", ".github/workflows/ci.yml", ".streamlit/config.toml",
                 "requirements.txt", "requirements-dev.txt", "requirements-tested.txt"):
        assert (ROOT / path).exists(), path


@pytest.mark.parametrize("module", ["src.train", "src.predict"])
def test_documented_cli_entry_points_work_from_any_directory(module, tmp_path):
    out = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0 and "usage" in out.stdout.lower()


def test_paths_are_absolute_and_independent_of_cwd(tmp_path):
    code = "from src import config; print(config.DATA_PATH); print(config.MODEL_DIR); print(config.RESULTS_DIR)"
    env = {k: v for k, v in os.environ.items() if not k.startswith("BEYONDACCURACY_")}
    env["PYTHONPATH"] = str(ROOT)
    from_elsewhere = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True,
                                    text=True, env=env).stdout.split()
    from_root = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                               text=True, env=env).stdout.split()
    assert from_elsewhere == from_root and len(from_root) == 3
    assert all(Path(p).is_absolute() for p in from_root)
    assert Path(from_root[0]) == ROOT / "data" / "creditcard.csv"


def test_gitignore_excludes_dataset_models_and_secrets():
    text = (ROOT / ".gitignore").read_text()
    for pattern in ("data/*.csv", "models/*.pkl", ".env", ".venv/", "__pycache__/"):
        assert pattern in text


def test_no_dataset_or_model_binary_is_tracked():
    tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout")
    files = tracked.stdout.split()
    assert not [f for f in files if f.endswith((".pkl", ".joblib")) or f.startswith("data/") and f.endswith(".csv")]
