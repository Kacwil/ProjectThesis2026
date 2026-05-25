import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_terrain_heatmap(terrain_hitcount: np.ndarray,
                         xs: np.ndarray | None = None,
                         ys: np.ndarray | None = None,
                         log_scale: bool = True,
                         cmap: str = "hot"):
    """
    Plot terrain hitcount heatmap.
    """

    heat = terrain_hitcount.copy()

    if log_scale:
        heat = np.log10(heat + 1)

    plt.figure(figsize=(8, 6))

    if xs is not None and ys is not None:
        extent = [xs[0], xs[-1], ys[0], ys[-1]]
        plt.imshow(
            heat,
            origin="lower",
            extent=extent,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap
        )
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        plt.imshow(
            heat,
            origin="lower",  # ensures y increases upward
            aspect="auto",
            interpolation="nearest",
            cmap=cmap
        )
        plt.xlabel("x index")
        plt.ylabel("y index")
    
    plt.colorbar(label="Hit count")
    plt.title("Terrain Hit Heatmap")
    plt.tight_layout()
    plt.show()