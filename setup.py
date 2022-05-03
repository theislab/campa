from pathlib import Path

from setuptools import setup, find_packages

try:
    from squidpy import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "Theislab & Pelkmanslab"
    __email__ = ", ".join(
        [
            "hannah.spitzer@helmholtz-muenchen.de",
            # TODO add scott here
        ]
    )
    __version__ = "0.0.1"

setup(
    name="campa",
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    packages=find_packages(),
    package_dir={"campa": "campa"},
    include_package_data=True,
    extras_require=dict(
        dev=["pre-commit>=2.9.0"],
        test=[
            "tox>=3.20.1",
            "pytest",
            "pytest-cov",
            "pytest-dependency",
            # "nbconvert",
            # "ipykernel",
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
