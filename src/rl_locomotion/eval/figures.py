from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np


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


def _fit_camera(model: Any, data: Any, margin: float = 1.25) -> tuple[np.ndarray, float]:
    """Centre and viewing distance that fit the *robot* in frame.

    `model.stat.extent` describes the whole scene, floor included, so using it
    frames short robots too loosely and crops tall ones. Measuring the robot
    works across morphologies — Go1 and Apollo differ by a factor of three in
    height.

    Measured over *geoms* rather than body origins, and inflated by each geom's
    bounding radius: a body origin sits at a joint, while the shell around it
    can extend a long way further. Ignoring that clips the tallest part of a
    robot out of frame.

    Returns (lookat, distance).
    """
    # Keep only geoms attached to a real body. Terrain — a plane, or a height
    # field spanning tens of metres — hangs off the world body (id 0), and
    # including it pushes the camera so far back the robot vanishes.
    keep = np.flatnonzero(model.geom_bodyid != 0)
    if keep.size == 0:
        return data.subtree_com[0], model.stat.extent * 2.0

    pos = data.geom_xpos[keep]
    rbound = model.geom_rbound[keep].reshape(-1, 1)
    lo = (pos - rbound).min(axis=0)
    hi = (pos + rbound).max(axis=0)

    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0

    fovy = np.deg2rad(model.vis.global_.fovy)
    distance = margin * radius / np.tan(fovy / 2.0)
    return centre, distance


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
    headlight: tuple[float, float] | None = None,
) -> np.ndarray:
    """One high-resolution frame from a specified camera.
    """

    model, data = _model_and_data(env, state)

    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    model.vis.quality.offsamples = samples

    if headlight is not None:
        # Scenes disagree about lighting. The flat-terrain scenes set a
        # headlight explicitly (diffuse .8, ambient .2); the rough-terrain ones
        # fall back on MuJoCo's dimmer default (.4/.1) and, with a dark rock
        # texture, print too dark to read.
        #
        # This raises a dim scene to the given floor and never lowers a bright
        # one — forcing a single value on every scene overexposes the ones that
        # were already lit correctly.
        diffuse, ambient = headlight
        if model.vis.headlight.diffuse[0] < diffuse:
            model.vis.headlight.diffuse = [diffuse] * 3
        if model.vis.headlight.ambient[0] < ambient:
            model.vis.headlight.ambient = [ambient] * 3

    fit_centre, fit_distance = _fit_camera(model, data, margin=zoom)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = azimuth
    camera.elevation = elevation
    camera.distance = distance if distance is not None else fit_distance
    camera.lookat[:] = lookat if lookat is not None else fit_centre

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


def reset_state(env: Any, seed: int = 0) -> Any:
    """The environment's own starting state, as `reset()` produces it.

    Worth the extra step over the model's default pose: the two differ a lot
    for some tasks. `Go1Getup` resets to a robot lying on its back, which the
    default pose does not show at all. Reset is randomised, so the seed is
    fixed to keep figures reproducible.
    """
    import jax

    return jax.jit(env.reset)(jax.random.PRNGKey(seed))


def save_all_env_figures(
    out_dir: str | Path,
    suite: str = "locomotion",
    env_names: list[str] | None = None,
    seed: int = 0,
    width: int = 800,
    height: int = 700,
    zoom: float = 1.25,
    headlight: tuple[float, float] | None = (0.8, 0.2),
    verbose: bool = True,
    **kwargs: Any,
) -> dict[str, Path]:
    """Render every environment of a suite at its own reset state.

    One image per *environment*, not per robot: this is what shows the
    difference between flat and rough terrain, and between a joystick task and
    a getup task on the same machine.

    Building 19 models takes a few minutes on CPU; `verbose` reports progress
    so a long run does not look hung.
    """
    from mujoco_playground import registry as pg_registry

    from rl_locomotion.envs.registry import list_envs

    if env_names is None:
        env_names = [info.name for info in list_envs(suite)]

    out_dir = Path(out_dir)
    written: dict[str, Path] = {}

    for i, name in enumerate(env_names, start=1):
        env = pg_registry.load(name)
        frame = hero_shot(
            env, reset_state(env, seed), width=width, height=height,
            zoom=zoom, headlight=headlight, **kwargs
        )
        written[name] = save_image(frame, out_dir / f"{name}.png")
        if verbose:
            print(f"[{i}/{len(env_names)}] {name}", flush=True)

    return written
