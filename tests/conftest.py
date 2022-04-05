import pytest


@pytest.fixture(autouse=True)
def _execute_from_tests_dir():
    # TODO this does not work as expected, still need to test from pytest dir
    # ensure that current dir is tests dir
    # reload constants to have correct EXPERIMENT_DIR
    import os
    import importlib

    from campa.constants import SCRIPTS_DIR
    import campa

    os.chdir(os.path.join(SCRIPTS_DIR, "tests"))
    importlib.reload(campa.constants)
