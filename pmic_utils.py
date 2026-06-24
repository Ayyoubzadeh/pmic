import numpy as np


class FeatureTransformer:
    """Picklable wrapper around the fitted variance / correlation / MI selectors."""
    def __init__(self, sel_vt, keep, sel_mi):
        self.sel_vt = sel_vt
        self.keep   = keep
        self.sel_mi = sel_mi

    def __call__(self, X):
        out = self.sel_vt.transform(X).astype(np.float32)[:, self.keep]
        if self.sel_mi is not None:
            out = self.sel_mi.transform(out)
        return out
