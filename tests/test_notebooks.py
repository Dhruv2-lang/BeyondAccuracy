"""Notebook integrity: valid JSON/nbformat, no stale errors, and executable end-to-end."""

import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parent.parent / "notebooks").glob("*.ipynb"))
REBUILT = ["03_imbalance_handling", "04_evaluation", "05_finalize_model"]


def test_notebooks_are_present():
    assert [p.stem for p in NOTEBOOKS][:5] == [
        "01_eda", "02_baseline", "03_imbalance_handling", "04_evaluation", "05_finalize_model"]


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_is_valid_json_and_nbformat(path):
    json.loads(path.read_text(encoding="utf-8"))            # catches concatenated / truncated JSON
    nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.stem)
def test_notebook_has_no_error_outputs_or_local_paths(path):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            assert not [o for o in cell.get("outputs", []) if o.output_type == "error"]
            assert "/Users/" not in cell.source and "C:\\" not in cell.source


@pytest.mark.parametrize("name", REBUILT)
def test_rebuilt_notebooks_ship_without_fabricated_outputs(name):
    nb = nbformat.read(path_of(name), as_version=4)
    assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code")


def path_of(name):
    return next(p for p in NOTEBOOKS if p.stem == name)


@pytest.mark.parametrize("name", REBUILT)
def test_rebuilt_notebooks_execute_top_to_bottom(name, monkeypatch, tmp_path, synthetic_csv, artifact_dir):
    """Run on synthetic data with all outputs redirected away from the repository."""
    pytest.importorskip("ipykernel")
    NotebookClient = pytest.importorskip("nbclient").NotebookClient
    monkeypatch.setenv("BEYONDACCURACY_DATA_PATH", str(synthetic_csv))
    monkeypatch.setenv("BEYONDACCURACY_MODEL_DIR", str(artifact_dir))
    monkeypatch.setenv("BEYONDACCURACY_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("BEYONDACCURACY_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("BEYONDACCURACY_QUICK", "1")
    nb = nbformat.read(path_of(name), as_version=4)
    NotebookClient(nb, timeout=300, kernel_name="python3",
                   resources={"metadata": {"path": str(path_of(name).parent)}}).execute()
