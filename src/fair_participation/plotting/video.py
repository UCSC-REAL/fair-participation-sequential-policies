import os
import matplotlib.animation as animation
from matplotlib import pyplot as plt

from fair_participation.base_logger import logger
from fair_participation.utils import PROJECT_ROOT


def video_filename(filename):
    return os.path.join(PROJECT_ROOT, "mp4", f"{filename}.mp4")


class Video:
    """
    Use a matplotlib figure to make a video.
    For each frame must:
      1. draw to figure
      2. call the `video.draw` method
      3. clear the figure/axes/Artists

    Example:

    fig, ax = plt.subplots(figsize=(6, 6))

    with Video('video_name', fig) as video:
        for _ in range(num_frames):
            render_to_fig()
            video.draw()
            ax.cla()
    """

    def __init__(
        self,
        filename: str,
        figure: plt.Figure,
        fps: int = 15,
        dpi: int = 100,
    ):
        self.video_file = video_filename(filename)
        self.writer = animation.FFMpegWriter(
            fps=fps, metadata={"title": filename, "artist": "Matplotlib"}
        )
        self.figure = figure
        self.dpi = dpi

    def __enter__(self):
        self.writer.setup(self.figure, self.video_file, dpi=self.dpi)
        return self

    def draw(self):
        # draw figure and clear axes
        self.writer.grab_frame()
        # TODO check clear axes
        # for ax in self.figure.axes:
        #     ax.cla()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.writer.finish()
        if exc_type is None:
            logger.info(f"Writing video to {self.video_file}.")
        else:
            logger.info(f"Removing file at {self.video_file}.")
            os.remove(self.video_file)
