from campa import data, pl, tl

__author__ = __maintainer__ = "Theislab & Pelkmanslab"
__email__ = ", ".join(
    [
        "hannah.spitzer@helmholtz-muenchen.de",
        # TODO add scott
    ]
)
__version__ = "0.0.1"


from importlib.metadata import version

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse