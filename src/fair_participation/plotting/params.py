import numpy as np

import seaborn as sns
from matplotlib.colors import to_rgba as to_rgba
import matplotlib.path as mpath

base_sns_theme_kwargs = {
    "style": "white",
    "context": "paper",
    "font": "serif",
}
base_rcparams = {
    "text.usetex": True,
    "axes.labelsize": 20.0,
    "xtick.labelsize": 18.0,
    "ytick.labelsize": 18.0,
    "axes.titlesize": 20.0,
    "legend.fontsize": 16.5,
    "xtick.bottom": True,
    "ytick.left": True,
    "lines.linewidth": 3,
}

cb_palette = sns.color_palette("colorblind")


_star = mpath.Path.unit_regular_star(6)
_circle = mpath.Path.unit_circle()
_cut_star = mpath.Path(
    vertices=np.concatenate([_circle.vertices, _star.vertices[::-1, ...]]),
    codes=np.concatenate([_circle.codes, _star.codes]),
)

_method_markers = {
    "Initialization": {
        "marker": "s",
        "color": to_rgba(cb_palette[7], alpha=0.8),
        "s": 300,
    },
    "RRM": {
        "marker": "D",
        "color": to_rgba(cb_palette[2], alpha=0.8),
        "s": 240,
    },
    "MPG": {
        "marker": "o",
        "color": to_rgba(cb_palette[4], alpha=0.8),
        "s": 240,
    },
    "CPG": {
        "marker": _cut_star,
        "color": to_rgba(cb_palette[1], alpha=0.9),
        "s": 260,
    },
}


def marker_map(key: str) -> dict:
    return {k: v[key] for k, v in _method_markers.items()}
