from pathlib import Path

from setuptools import setup, find_packages

try:
    from campa import __email__, __author__, __version__, __maintainer__
except ImportError:
    __author__ = __maintainer__ = "Hannah Spitzer, Scott Berry"
    __email__ = ", ".join(
        [
            "hannah.spitzer@helmholtz-muenchen.de",
            "scott.berry@unsw.edu.au"
        ]
    )
    __version__ = "0.0.3"

setup(
    name="campa",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    version=__version__,
    author=__author__,
    author_email=__email__,
    maintainer=__author__,
    maintainer_email=__email__,
    description=Path("README_pypi.rst").read_text("utf-8").splitlines()[2],
    long_description=Path("README_pypi.rst").read_text("utf-8"),
    long_description_content_type="text/x-rst; charset=UTF-8",
    url="https://github.com/theislab/campa",
    download_url="https://pypi.org/project/campa/",
    project_urls={
        "Documentation": "https://campa.readthedocs.io/en/stable",
        "Source Code": "https://github.com/theislab/campa",
    },
    license="BSD",
    platforms=["Linux", "MacOSX"],
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
