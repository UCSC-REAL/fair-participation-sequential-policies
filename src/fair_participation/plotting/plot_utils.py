from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
from numpy.typing import NDArray, ArrayLike

from scipy.spatial import ConvexHull
from numpy.linalg import det
import jax
from fair_participation.base_logger import logger
from matplotlib.colors import LightSource

import mpl_toolkits.mplot3d as a3

mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

font = {"size": 13}
mpl.rc("font", **font)


def savefig(filename, **kwargs):
    logger.info(f"Saving {filename}")
    plt.savefig(filename, **kwargs)


class UpdatingPlot:
    def update(self, **_):
        raise NotImplementedError


def sample_hull_uniform(hull: ConvexHull, n: int, seed: int = 0) -> ArrayLike:
    """
    Uniformly sample points from surface of a convex hull.

    :param hull: Convex hull.
    :param n: Number of points to sample.
    :param seed: Random seed.
    :return: Array of points.
    """
    key = jax.random.PRNGKey(seed)
    dim = hull.points.shape[-1]
    faces = hull.points[hull.simplices]

    vols = np.abs(det(faces)) / np.math.factorial(dim)

    key, subkey = jax.random.split(key, 2)
    which_face = jax.random.choice(key, len(vols), shape=(n,), p=(vols / vols.sum()))
    convex_comb = jax.random.dirichlet(subkey, np.ones((n, dim)))

    return np.einsum("ijk, ij -> ik", faces[which_face], convex_comb)


def upsample_triangles(
    points: ArrayLike, faces: ArrayLike, normals: ArrayLike
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Upsample a triangulation by adding a new point at the center of each triangle.

    :param points: Array of points.
    :param faces: Array of faces.
    :param normals: Array of normals.
    :return: Upsampled points, faces, and normals.
    """
    new_points = list(points)
    new_faces = []
    new_normals = []

    idx = len(points)

    for i, face in enumerate(faces):
        a, b, c = points[face]

        # 4 new points
        d = (a + b) / 2
        e = (b + c) / 2
        f = (a + c) / 2

        new_points.extend([d, e, f])
        d_idx, e_idx, f_idx = idx, idx + 1, idx + 2
        idx += 3

        # 6 new faces
        a_idx, b_idx, c_idx = face

        new_faces.extend(
            [
                (a_idx, d_idx, f_idx),
                (d_idx, e_idx, f_idx),
                (d_idx, b_idx, e_idx),
                (f_idx, e_idx, c_idx),
            ]
        )

        # 6 new normals
        new_normals.extend([normals[i]] * 4)

    return np.array(new_points), np.array(new_faces), np.array(new_normals)


def upsample_hull_3d(
    points: ArrayLike, deg: int = 1
) -> tuple[NDArray, NDArray, NDArray]:
    """
    Upsample a convex hull in 3d.

    :param points: Array of points.
    :param deg: Degree of upsampling.
    :return: Upsampled points, faces, and normals.
    """
    hull = ConvexHull(points)

    points = hull.points
    faces = hull.simplices
    normals = hull.equations[:, :-1]

    for _ in range(deg):
        points, faces, normals = upsample_triangles(points, faces, normals)

    return points, faces, normals


def inclusive_hull_order_2d(points_2d: ArrayLike):
    """
    O(n^2) insert method like insertion sort.
    Given a list of points known to lie on a convex hull in 2d, order them counterclockwise.

    :param points_2d: List of points.
    :return: Ordered list of points.
    """

    hull = ConvexHull(points_2d)
    vertices = hull.vertices
    out = list(hull.points[vertices])
    left_out = list(set(range(len(points_2d))) - set(vertices))
    idx = 0

    while left_out:
        p = hull.points[left_out.pop()]
        for idx in range(len(out) - 1):
            d = p - out[idx]
            r = out[idx + 1] - out[idx]
            # p as convex combination b/w out[i] and out[i+1]
            interp = np.dot(d, r) / np.dot(r, r)
            # distance from line b/w out[i] and out[i+1]
            proj = d - interp * r
            dist = np.dot(proj, proj)
            if (dist < 1e-9) and (0 < interp < 1):
                break
        out.insert(idx + 1, p)

    return out


def project_hull(
    points: ArrayLike, val: float, dim: int = 0, wrap: bool = True
) -> NDArray:
    """
    Project a convex hull onto the plane (x_dim = val).

    :param points: Array of points.
    :param val: Value to project onto.
    :param dim: Dimension to project onto.
    :param wrap: Whether to wrap the hull around.
    :return: Projected points.
    """
    # eliminate chosen dimension and replace with val
    low_dim_points = np.delete(points, dim, 1)
    proj_hull = ConvexHull(low_dim_points)
    proj_points = proj_hull.points[proj_hull.vertices]
    out = np.insert(proj_points, dim, val, axis=1)
    if not wrap:
        return out
    # unify first and last point
    return np.insert(out, len(out), out[0, :], axis=0)


def get_normal(triangle_3d):
    a, b, c = triangle_3d
    out = np.cross(b - a, c - a)
    return out / np.sqrt(np.dot(out, out))


def plot_triangles(ax, triangles, normals):
    tri = a3.art3d.Poly3DCollection(triangles, zorder=1)
    ls = LightSource(azdeg=60, altdeg=60.0)
    rgb = np.array([0.0, 0.0, 1.0, 0.5])
    rgbt = np.array([0.0, 0.0, 0.0, 0.0])
    tri.set_facecolor(
        np.array([shade * rgb for shade in ls.shade_normals(normals, fraction=1.0)])
    )
    tri.set_edgecolor(
        np.array([shade * rgbt for shade in ls.shade_normals(normals, fraction=1.0)])
    )
    ax.add_collection3d(tri)


def set_corner_ticks(ax, axes: str):
    """Set ticks only at the corners of the current axes.

    :param ax: Matplotlib axis.
    :param axes: String of axes to set ticks on. Should be a subset of "xyz".
    """
    if "x" in axes:
        ax.set_xticks(ax.get_xlim())
    if "y" in axes:
        ax.set_yticks(ax.get_ylim())
    if "z" in axes:
        ax.set_zticks(ax.get_zlim())


def set_nice_limits(
    ax: mpl.axes,
    clip_min: float | ArrayLike,
    clip_max: float | ArrayLike,
    res: float = 0.02,
    equal_aspect: bool = True,
):
    """For an axis with equal box aspect ratio, set the limits of the current axes to be the closest square.

    If equal_aspect is True, ensure that the aspect ratio of the current axes is equal, not just the box.
    """
    old_ax = plt.gca()
    plt.sca(ax)
    lims = []
    if isinstance(clip_min, (int, float)):
        clip_min = np.array([clip_min, clip_min])
    if isinstance(clip_max, (int, float)):
        clip_max = np.array([clip_max, clip_max])
    for plt_lim, cmin, cmax in zip((plt.xlim, plt.ylim), clip_min, clip_max):
        lim = plt_lim()
        lim = np.array([np.floor(lim[0] / res), np.ceil(lim[1] / res)]) * res
        lims.append(np.clip(lim, cmin, cmax))
    lims = np.array(lims)
    if equal_aspect:
        # Since we need equal aspect, keep the larger range
        lim_range = np.max(lims[:, 1] - lims[:, 0])
        lims[0, 1] = lims[0, 0] + lim_range
        lims[1, 1] = lims[1, 0] + lim_range
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.sca(old_ax)


def use_two_ticks(ax, axis: str, others: Optional[list] = None):
    if axis == "x":
        getticks, setticks, getlim = ax.get_xticks, ax.set_xticks, ax.get_xlim
    elif axis == "y":
        getticks, setticks, getlim = ax.get_yticks, ax.set_yticks, ax.get_ylim
    elif axis == "z":
        getticks, setticks, getlim = ax.get_zticks, ax.set_zticks, ax.get_zlim
    else:
        raise ValueError
    if others is None:
        others = []
    # TODO this was not right: ticks can be outside the axes
    min_ = min(tick for tick in getticks() if tick > getlim()[0])
    max_ = max(tick for tick in getticks() if tick < getlim()[1])
    setticks([min_, max_] + others)
