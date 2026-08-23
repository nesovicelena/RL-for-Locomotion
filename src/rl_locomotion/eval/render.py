"""Turn trajectories into pictures and video.

Rendering needs an OpenGL backend, chosen by the `MUJOCO_GL` environment
variable **before** MuJoCo is first imported:

* laptop  -> `glfw`
* pod (headless) -> `egl`

`configure_backend()` sets a sensible default, but only has an effect if it
runs before the first MuJoCo import — call it at the top of a notebook.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rl_locomotion.eval.rollout import Rollout


def configure_backend(backend: str | None = None) -> str:
    """Set `MUJOCO_GL` if unset. Returns the backend in effect.

    Must run before MuJoCo is imported to take effect; if MuJoCo is already
    loaded the existing backend stays and this is a no-op report.
    """
    if backend is None:
        backend = os.environ.get(
            "MUJOCO_GL", "glfw" if platform.system() == "Darwin" else "egl"
        )
    os.environ.setdefault("MUJOCO_GL", backend)
    return os.environ["MUJOCO_GL"]


def render_rollout(
    env: Any,
    rollout: Rollout,
    height: int = 240,
    width: int = 320,
    camera: str | None = None,
    every: int = 1,
) -> np.ndarray:
    """Frames for a rollout, as a `(n_frames, height, width, 3)` uint8 array.

    `every > 1` renders a subsample, which is the cheap way to preview a long
    episode. Pass the same value to `video_fps` to keep playback real-time.
    """
    frames = env.render(
        rollout.states[::every], height=height, width=width, camera=camera
    )
    return np.asarray(frames)


def render_state(
    env: Any,
    state: Any,
    height: int = 240,
    width: int = 320,
    camera: str | None = None,
) -> np.ndarray:
    """A single frame — the standard way to eyeball a robot's starting pose."""
    return np.asarray(env.render([state], height=height, width=width, camera=camera))[0]


def video_fps(env: Any, every: int = 1) -> float:
    """Playback rate that makes video real-time for this environment."""
    return 1.0 / (float(env.dt) * every)


def save_video(
    frames: np.ndarray, path: str | Path, fps: float = 30.0
) -> Path:
    """Write frames to an mp4. Creates parent directories."""
    import mediapy as media

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(str(path), frames, fps=fps)
    return path


def show_video(frames: np.ndarray, fps: float = 30.0, **kwargs: Any) -> Any:
    """Inline playback in a notebook."""
    import mediapy as media

    return media.show_video(frames, fps=fps, **kwargs)


def gallery(
    frames: Sequence[np.ndarray],
    titles: Sequence[str] | None = None,
    ncols: int = 4,
    figsize_per_tile: float = 3.0,
) -> Any:
    """Grid of stills — for comparing several robots side by side."""
    import matplotlib.pyplot as plt

    n = len(frames)
    if n == 0:
        raise ValueError("nothing to show: `frames` is empty")

    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceiling division
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * figsize_per_tile, nrows * figsize_per_tile),
        squeeze=False,
    )

    for ax in axes.flat:
        ax.axis("off")
    for i, frame in enumerate(frames):
        ax = axes[i // ncols][i % ncols]
        ax.imshow(frame)
        if titles is not None:
            ax.set_title(titles[i], fontsize=9)

    fig.tight_layout()
    return fig


def plot_rollout(rollout: Rollout, figsize: tuple[float, float] = (10.0, 3.0)) -> Any:
    """Reward and action traces over an episode.

    Worth a glance next to the video: a reward series that collapses partway
    through usually corresponds to the moment the robot falls.
    """
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_a) = plt.subplots(1, 2, figsize=figsize)
    t = np.arange(rollout.n_steps) * rollout.ctrl_dt

    ax_r.plot(t, rollout.rewards, lw=1)
    ax_r.set(xlabel="time (s)", ylabel="reward", title="reward per step")
    ax_r.grid(alpha=0.3)

    ax_a.plot(t, rollout.actions, lw=0.7, alpha=0.7)
    ax_a.set(xlabel="time (s)", ylabel="action", title=f"actions ({rollout.actions.shape[1]} dof)")
    ax_a.grid(alpha=0.3)

    fig.tight_layout()
    return fig


def plot_reward_terms(
    rollout: Rollout, top_n: int = 8, figsize: tuple[float, float] = (9.0, 4.0)
) -> Any:
    """Mean contribution of each reward term over an episode.

    Most locomotion tasks clip the summed reward at zero (see
    `Rollout.metrics_frame`). When the total sits flat at zero, this is the plot
    that says why — which penalties are dominating.
    """
    import matplotlib.pyplot as plt

    df = rollout.metrics_frame()
    reward_cols = [c for c in df.columns if c.startswith("reward/")]
    if not reward_cols:
        raise ValueError("this rollout recorded no reward/* metrics")

    means = df[reward_cols].mean()
    means.index = [c.removeprefix("reward/") for c in means.index]
    means = means.reindex(means.abs().sort_values(ascending=False).index)[:top_n]

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:red" if v < 0 else "tab:green" for v in means.values]
    ax.barh(means.index[::-1], means.values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", lw=0.8)
    ax.set(xlabel="mean per-step contribution", title="reward terms")
    ax.grid(alpha=0.3, axis="x")

    fig.tight_layout()
    return fig
