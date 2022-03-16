import numpy as np


def pad_border(arr, mode=None):
    """
    pad border pixels.

    Args:
        mode: None - do nothing and return arr
              'center' - replace zeros with value from center pixel
    Returns:
        modified arr
    """
    if mode is None:
        return arr
    elif mode == "center":
        s = arr.shape[0]
        center = arr[s // 2, s // 2]
        # pixels to mask
        mask = np.all(arr == 0, axis=2)
        arr[mask] = center
        return arr
    else:
        raise NotImplementedError(f"pad_border: mode={mode}")


class BoundingBox:
    def __init__(self, ymin, xmin, h, w):
        self.ymin = ymin
        self.xmin = xmin
        self.h = h
        self.w = w

    @classmethod
    def from_corners(cls, arr):
        assert len(arr) == 4, "need 4 coordinates for BoundingBox"
        ymin, xmin, ymax, xmax = arr
        h = ymax - ymin
        w = xmax - xmin
        return cls(ymin, xmin, h, w)

    @classmethod
    def from_center(cls, yc, xc, h, w):
        return cls(yc - h / 2, xc - w / 2, h, w)

    @property
    def ymax(self):
        return self.ymin + self.h

    @property
    def xmax(self):
        return self.xmin + self.w

    @property
    def center(self):
        return self.ymin + self.h / 2, self.xmin + self.w / 2

    @property
    def shape(self):
        return (self.h, self.w)

    def __str__(self):
        return "BoundingBox at ({},{}) with shape ({},{})".format(
            self.ymin, self.xmin, self.h, self.w
        )

    def __repr__(self):
        return f"BoundingBox({self.ymin},{self.xmin},{self.h},{self.w})"

    def as_int(self):
        return BoundingBox(
            int(round(self.ymin)),
            int(round(self.xmin)),
            int(round(self.h)),
            int(round(self.w)),
        )

    def fit_in_shape(self, shape):
        ah, aw = shape[:2]
        yoff1 = max(-self.ymin, 0)
        yoff2 = max(self.ymax - ah, 0)
        xoff1 = max(-self.xmin, 0)
        xoff2 = max(self.xmax - aw, 0)
        fitted_bbox = BoundingBox.from_corners(
            [
                max(0, self.ymin),
                max(0, self.xmin),
                min(self.ymax, ah),
                min(self.xmax, aw),
            ]
        )
        return fitted_bbox, yoff1, yoff2, xoff1, xoff2

    def crop(self, arr):
        assert len(arr.shape) >= 2, "arr needs to be at least 2d"
        bbox = self.as_int()
        res = np.zeros(bbox.shape + arr.shape[2:], dtype=arr.dtype)
        bbox_crop, yoff1, yoff2, xoff1, xoff2 = bbox.fit_in_shape(arr.shape)
        res[yoff1 : bbox.h - yoff2, xoff1 : bbox.w - xoff2] = arr[
            bbox_crop.ymin : bbox_crop.ymax, bbox_crop.xmin : bbox_crop.xmax
        ]
        return res
