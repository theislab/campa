from pathlib import Path

from nbconvert.preprocessors import ExecutePreprocessor
import pytest
import nbformat

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize(
    "notebook",
    [
        pytest.param("mpp_data.ipynb", marks=pytest.mark.dependency(name="mpp_data")),
        pytest.param(
            "nn_dataset.ipynb",
            marks=pytest.mark.dependency(name="nn_dataset", depends=["mpp_data"]),
        ),
    ],
)
def test_notebook_exec(notebook):
    with open(ROOT / "notebooks" / notebook) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        try:
            assert ep.preprocess(nb) is not None, f"Got empty notebook for {notebook}"
        except Exception:  # noqa: B902
            pytest.fail(f"Failed executing {notebook}")
