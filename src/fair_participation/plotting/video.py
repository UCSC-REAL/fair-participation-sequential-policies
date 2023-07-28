import matplotlib.animation as animation
from matplotlib import pyplot as plt

from fair_participation.base_logger import logger


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
        render_flag: bool = True,
        fps: int = 15,
        dpi: int = 100,
    ):
        self.video_file = f"./mp4/{filename}.mp4"

        # whether to actually do anything
        self.render_flag = render_flag

        if render_flag:
            self.writer = animation.FFMpegWriter(
                fps=fps, metadata={"title": filename, "artist": "Matplotlib"}
            )

        self.figure = figure
        self.dpi = dpi

    def __enter__(self):
        if self.render_flag:
            # make file
            self.writer.setup(self.figure, self.video_file, dpi=self.dpi)

        return self

    def draw(self):
        if self.render_flag:
            # draw figure and clear axes
            self.writer.grab_frame()
            for ax in self.figure.axes:
                ax.cla()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.render_flag:
            # finalize file
            self.writer.finish()
            logger.info(f"Writing video to {self.video_file}.")
