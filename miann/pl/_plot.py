# --- Plotting functions for evaluating experiments ---
import numpy as np

# TODO remove? TODO don't think its needed anymore
def map_img(imgs, value_map=None):
    """
    maps list of images containing map keys to map values.
    Used for displaying cluster images. If map is None, create and return a default one
    """
    if value_map is None:
        vals = sorted(list(set(np.concatenate([np.unique(i) for i in imgs]))))
        value_map = {val: i for i, val in enumerate(vals)}
    outs = []
    for img in imgs:
        out = np.zeros(img.shape)
        for key, val in value_map.items():
            out[img==key] = val
        outs.append(out)
    return outs, value_map


def annotate_img(img, annotation=None, from_col='clustering', to_col=None, color=False):
    """
    Args:
        img: image to annotate
        annotation: pd.DataFrame containing annotation (from Cluster)
        from_col: column containing current values in image
        to_col: column containing desired mapping. If None, use from_col.
        color: if True, use column to_col+"_colors" to get colormap and color image
    """
    if to_col is None:
        to_col = from_col
    if color:
        to_col = to_col + '_colors'
        res = np.zeros(img.shape + (3,), dtype=np.uint8)
    else:
        if from_col == to_col:
            # no need to change anything
            return img
        res = np.zeros_like(img)
    for _, row in annotation.iterrows():
        to_value = row[to_col]
        if color:
            to_value = hex2rgb(to_value)
        res[img == row[from_col]] = to_value
    return res.squeeze() if color else res

def hex2rgb(h):
    h = h.lstrip('#')
    return list(int(h[i:i+2], 16) for i in (0, 2, 4))