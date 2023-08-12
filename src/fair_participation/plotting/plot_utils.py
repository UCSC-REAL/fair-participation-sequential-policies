import matplotlib as mpl

import numpy as np

from scipy.spatial import ConvexHull, Delaunay
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


def savefig(fig, filename):
    logger.info(f"Saving {filename}")
    fig.savefig(filename)


class UpdatingPlot:
    def update(state, **_):
        pass


def sample_hull_uniform(hull, n, seed=0):
    """
    Uniformly sample points from surface of a convex hull
    """
    key = jax.random.PRNGKey(seed)
    dim = hull.points.shape[-1]
    faces = hull.points[hull.simplices]

    vols = np.abs(det(faces)) / np.math.factorial(dim)

    key, subkey = jax.random.split(key, 2)
    which_face = jax.random.choice(key, len(vols), shape=(n,), p=(vols / vols.sum()))
    convex_comb = jax.random.dirichlet(subkey, np.ones((n, dim)))

    return np.einsum("ijk, ij -> ik", faces[which_face], convex_comb)


def upsample_triangles(points, faces, normals):
    """
    accept faces as array of indices of points

    points: array
    faces: array
    normals: array

    give back new array of points,
    new array of faces,
    new array of normals.
    """
    new_points = list(points)
    new_faces = []
    new_normals = []

    idx = len(points)

    for (i, face) in enumerate(faces):

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


def upsample_hull_3d(points, deg=1):
    hull = ConvexHull(points)

    points = hull.points
    faces = hull.simplices
    normals = hull.equations[:, :-1]

    for _ in range(deg):
        points, faces, normals = upsample_triangles(points, faces, normals)

    return points, faces, normals


# def sample_hull_3d(hull, deg):
#     faces = hull.points[hull.simplices]

#     upsampled_points = set(tuple(p) for p in hull.points)

#     for f in faces:
#         upsampled_points.update(tuple(p) for p in upsample_simplex_3d(list(f), deg))

#     return np.array(list(upsampled_points))


def inclusive_hull_order_2d(points_2d):
    """
    O(n^2) insert method like insertion sort

    Given a list of points known to lie on a convex hull in 2d,
    order them counterclockwise.
    """

    hull = ConvexHull(points_2d)
    vertices = hull.vertices
    out = list(hull.points[vertices])
    left_out = list(set(range(len(points_2d))) - set(vertices))

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


# def inclusive_triangulation_3d(points_3d):
#     """
#     return triangle indices and normal vectors of surface of convex shape in 3d

#     O(n^2)
#     """

#     points_3d = np.array(points_3d)
#     hull = ConvexHull(points_3d)
#     hull_normals = hull.equations[:, :-1]
#     hull_surfaces = hull.equations[:, -1]

#     faces = [list(s) for s in hull.simplices]

#     # indices of points not included in default triangulation
#     left_out = list(set(range(len(points_3d))) - set(hull.vertices))

#     # figure out which face each left-out point belongs to
#     while left_out:
#         p_idx = left_out.pop()
#         p = points_3d[p_idx]
#         f_idx = np.argmin(np.abs(np.einsum("ij,j->i", hull_normals, p) + hull_surfaces))
#         faces[f_idx].append(p_idx)

#     out = []
#     normals = []
#     for (i, f) in enumerate(faces):
#         if len(f) == 3:
#             out.append(f)
#             normals.append(hull_normals[i])
#             continue

#         # contruct orthogonal basis for plane of simplex
#         v = hull_normals[i]
#         a, b, c = v
#         if c < a:
#             x = np.array([b, -a, 0])
#         else:
#             x = np.array([0, -c, b])

#         # normalize vectors
#         x = x / np.sqrt(np.dot(x, x))
#         y = np.cross(v, x)
#         y = y / np.sqrt(np.dot(y, y))

#         # project onto local simplex
#         points_2d = [np.array([np.dot(x, z), np.dot(y, z)]) for z in points_3d[f]]

#         # generate triangles
#         deln = Delaunay(points_2d)

#         # get indices
#         simps = [list(l) for l in np.array(f)[deln.simplices]]

#         out.extend(simps)
#         normals.extend([hull_normals[i]] * len(simps))

#     return np.array(out), np.array(normals)


def project_hull(points, val, dim=0, wrap=True):
    """
    project a convex hull onto the plain (x_dim = val)
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


# TODO vectorize
def get_normal(triangle_3d):
    a, b, c = triangle_3d
    out = np.cross(b - a, c - a)
    return out / np.sqrt(np.dot(out, out))


def plot_triangles(ax, triangles, normals, **kwargs):

    tri = a3.art3d.Poly3DCollection(triangles)

    ls = LightSource(azdeg=60, altdeg=60.0)

    rgb = np.array([0.0, 0.0, 1.0, 0.7])

    tri.set_facecolor(
        np.array([shade * rgb for shade in ls.shade_normals(normals, fraction=1.0)])
    )

    ax.add_collection3d(tri)


# TODO not sure about these
def use_two_ticks_x(ax, others=[]):
    x = ax.get_xticks()
    ax.set_xticks([min(x), max(x)] + others)


def use_two_ticks_y(ax, others=[]):
    y = ax.get_yticks()
    ax.set_yticks([min(y), max(y)] + others)


def use_two_ticks_z(ax, others=[]):
    z = ax.get_zticks()
    ax.set_zticks([min(z), max(z)] + others)
