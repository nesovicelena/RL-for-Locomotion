from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import mujoco


def _model_and_data(env: Any, state: Any = None) -> tuple[Any, Any]:
    """An `MjModel` and a posed `MjData` for the given state.
    """
    import mujoco

    model = env.mj_model
    data = mujoco.MjData(model)

    if state is not None:
        data.qpos, data.qvel = state.data.qpos, state.data.qvel
        # mocap_pos and mocap_quat are not part of the state, but they are needed for rendering
        data.mocap_pos, data.mocap_quat = state.data.mocap_pos, state.data.mocap_quat
        # xfrc_applied is actually part of the state
        data.xfrc_applied = state.data.xfrc_applied

    mujoco.mj_forward(model, data)
    return model, data


def hero_shot(
    env: Any,
    state: Any = None,
    width: int = 1600,
    height: int = 1200, # 4:3 aspect ratio for a report figure
    azimuth: float = 135.0, # 45° from the front, 90° from the side, 135° from the back
    elevation: float = -20.0, # 20° below the horizon, to show the robot's feet and ground plane
    distance: float | None = None,
    zoom: float = 1.2,
    lookat: tuple[float, float, float] | None = None,
    samples: int = 8,
    shadows: bool = True,
    reflections: bool = True,
    skybox: bool = True,
) -> np.ndarray:
    """One high-resolution frame from a specified camera.
    """

    model, data = _model_and_data(env, state)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.quality.offsamples = samples

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = (
        distance if distance is not None else model.stat.extent * zoom
    )
    camera.lookat[:] = lookat if lookat is not None else data.subtree_com[0]

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        flags = renderer.scene.flags
        flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = reflections
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = skybox
        return renderer.render()
    finally:
        renderer.close()


def orbit(
    env: Any,
    state: Any = None,
    n_views: int = 4,
    start_azimuth: float = 45.0,
    **kwargs: Any,
) -> list[np.ndarray]:
    """`n_views` shots evenly spaced around the model.

    A four-view plate showing a robot from several sides reads far better in a
    report than a single angle.
    """
    step = 360.0 / n_views
    return [
        hero_shot(env, state, azimuth=start_azimuth + i * step, **kwargs)
        for i in range(n_views)
    ]


def save_image(frame: np.ndarray, path: str | Path) -> Path:
    """Write a frame to PNG. Creates parent directories."""
    import mediapy as media

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    media.write_image(str(path), frame)
    return path


def robot_lineup(
    env_names: list[str] | None = None,
    titles: list[str] | None = None,
    ncols: int = 3,
    figsize_per_tile: float = 4.0,
    width: int = 800,
    height: int = 640,
    **kwargs: Any,
) -> Any:
    """Every robot in one figure — a grid of hero shots at the reset pose.

    """
    from mujoco_playground import registry as pg_registry

    from rl_locomotion.eval.render import gallery

    if env_names is None:
        from rl_locomotion.envs.registry import list_models

        models = [m for m in list_models("locomotion") if m.envs]
        env_names = [m.envs[0] for m in models]
        titles = titles or [m.platform for m in models]
    if not env_names:
        raise ValueError("nothing to render: `env_names` is empty")

    frames = [
        hero_shot(pg_registry.load(name), width=width, height=height, **kwargs)
        for name in env_names
    ]
    return gallery(frames, titles or env_names, ncols=ncols,
                   figsize_per_tile=figsize_per_tile)


def save_robot_lineup(
    path: str | Path,
    env_names: list[str] | None = None,
    dpi: int = 200,
    **kwargs: Any,
) -> Path:
    """`robot_lineup` written straight to a file, closed afterwards.

    """
    import matplotlib.pyplot as plt

    figure = robot_lineup(env_names, **kwargs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)
    return path


def save_model_figures(
    env_names: list[str],
    out_dir: str | Path,
    prefix: str = "",
    **kwargs: Any,
) -> list[Path]:
    """Render one hero shot per environment and write them all to `out_dir`.

    The batch version, for regenerating every figure in the report after a
    camera or styling change.
    """
    from mujoco_playground import registry as pg_registry

    out_dir = Path(out_dir)
    paths = []
    for name in env_names:
        env = pg_registry.load(name)
        frame = hero_shot(env, **kwargs)
        paths.append(save_image(frame, out_dir / f"{prefix}{name}.png"))
    return paths
