import matplotlib as mpl

from fair_participation.base_logger import logger


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


# TODO not sure about these
def use_two_ticks_x(ax):
    x = ax.get_xticks()
    ax.set_xticks(x[:: len(x) - 1])


def use_two_ticks_y(ax):
    y = ax.get_yticks()
    ax.set_yticks(y[:: len(y) - 1])
