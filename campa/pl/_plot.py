# --- Plotting functions for evaluating experiments ---
from typing import Optional

import numpy as np
import pandas as pd


def annotate_img(
    img: np.ndarray,
    annotation: Optional[pd.DataFrame] = None,
    from_col: str = "clustering",
    to_col: Optional[str] = None,
    color: bool = False,
) -> np.ndarray:
    """
    Annotate cluster image.

    Parameters
    ----------
    img
        Image to annotate.
    annotation
        :attr:`Cluster.cluster_annotation` containing mapping of classes to cluster names and colours.
    from_col
        Annotation column containing current values in image.
    to_col
        Annotation column containing desired mapping. If None, use ``from_col``.
    color
        If True, use annotation column ``to_col+"_colors"`` to get colormap and colour image.
    Returns
    -------
    Annotated image.
    """
    if to_col is None:
        to_col = from_col
    if color:
        to_col = to_col + "_colors"
        res = np.zeros(img.shape + (3,), dtype=np.uint8)
    else:
        if from_col == to_col:
            # no need to change anything
            return img
        assert annotation is not None
        res = np.zeros_like(img, dtype=annotation[to_col].dtype)
    assert annotation is not None
    for _, row in annotation.iterrows():
        to_value = row[to_col]
        if color:
            to_value = hex2rgb(to_value)
        res[img == row[from_col]] = to_value
    return res.squeeze() if color else res


def hex2rgb(h):
    """Convert hex color string to rgb tuple."""
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]
