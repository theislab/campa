from pathlib import Path

from setuptools import setup, find_packages

setup(
    name="campa",
    version="0.1",
    packages=find_packages(),
    package_dir={"campa": "campa"},
    # data_files=[('campa', ['campa/config.ini.example'])],
    include_package_data=True,
    extras_require=dict(
        dev=["pre-commit>=2.9.0"],
        test=[
            "tox>=3.20.1",
            "pytest",
            "pytest-cov",
            "pytest-dependency",
            "nbconvert",
            "ipykernel",
        ],
        docs=[
            l.strip()
            for l in (Path("docs") / "requirements.txt").read_text("utf-8").splitlines()
            if not l.startswith("-r")
        ],
    ),
    install_requires=[line.strip() for line in Path("requirements.txt").read_text("utf-8").splitlines()],
    entry_points={
        "console_scripts": ["campa=campa.cli.main:CAMPA"],
    },
)
