"""Browse what MuJoCo Playground offers: environments and the robot models
behind them.
"""
from __future__ import annotations

import contextlib
import functools
import io
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import importlib
import pandas as pd
from mujoco_playground import registry
from mujoco_playground._src import mjx_env

SUITES = ("locomotion", "manipulation", "dm_control_suite")
_SUITE_ENVS = {
    "locomotion": "mujoco_playground._src.locomotion",
    "manipulation": "mujoco_playground._src.manipulation",
    "dm_control_suite": "mujoco_playground._src.dm_control_suite",
}


@dataclass(frozen=True)
class EnvInfo:
    """One registered environment."""

    name: str
    suite: str
    platform: str  # e.g. "go1", "franka_emika_panda"
    task: str  # e.g. "joystick", "getup"
    ctrl_dt: float
    sim_dt: float
    episode_length: int
    action_repeat: int
    has_domain_randomizer: bool

    @property
    def sim_steps_per_ctrl_step(self) -> int:
        return round(self.ctrl_dt / self.sim_dt)

    @property
    def episode_seconds(self) -> float:
        return self.episode_length * self.ctrl_dt


@dataclass(frozen=True)
class ModelInfo:
    """One robot / model family shipped with Playground."""

    platform: str
    suite: str
    n_envs: int
    envs: list[str] = field(default_factory=list)
    xmls: list[str] = field(default_factory=list)


def _source_module(env_name: str, suite: str) -> str:
    """Module path the env class was defined in, e.g. `...locomotion.go1.joystick`."""
    envs = importlib.import_module(_SUITE_ENVS[suite])._envs
    entry = envs[env_name]
    target = entry.func if isinstance(entry, functools.partial) else entry
    return getattr(target, "__module__", "")


def _split_module(module: str, suite: str) -> tuple[str, str]:
    """`...locomotion.go1.joystick` -> ("go1", "joystick")."""
    parts = module.split(".")
    if suite not in parts:
        return ("unknown", "unknown")
    tail = parts[parts.index(suite) + 1 :]
    if len(tail) >= 2:
        return (tail[0], ".".join(tail[1:]))
    if len(tail) == 1:
        return (tail[0], tail[0])
    return ("unknown", "unknown")


def _suite_of(env_name: str) -> str:
    for suite in SUITES:
        if env_name in getattr(registry, suite).ALL_ENVS:
            return suite
    raise KeyError(f"{env_name!r} is not a registered Playground environment")


def env_info(env_name: str) -> EnvInfo:
    """Metadata for one environment."""
    suite = _suite_of(env_name)
    platform, task = _split_module(_source_module(env_name, suite), suite)
    cfg = registry.get_default_config(env_name)

    try:
        # Upstream prints "does not have a domain randomizer" for every env
        # that lacks one. Expected here — we are probing on purpose — and it
        # would bury the table in noise.
        with contextlib.redirect_stdout(io.StringIO()):
            randomizer = registry.get_domain_randomizer(env_name)
    except (KeyError, ValueError):
        # Only locomotion and manipulation define randomizers at all.
        randomizer = None

    return EnvInfo(
        name=env_name,
        suite=suite,
        platform=platform,
        task=task,
        ctrl_dt=float(cfg.ctrl_dt),
        sim_dt=float(cfg.sim_dt),
        episode_length=int(cfg.episode_length),
        action_repeat=int(cfg.action_repeat),
        has_domain_randomizer=randomizer is not None,
    )


def list_envs(suite: str | Iterable[str] | None = None) -> list[EnvInfo]:
    """Every registered environment, optionally filtered to one or more suites."""
    if suite is None:
        suites: tuple[str, ...] = SUITES
    elif isinstance(suite, str):
        suites = (suite,)
    else:
        suites = tuple(suite)

    unknown = set(suites) - set(SUITES)
    if unknown:
        raise ValueError(f"unknown suite(s) {sorted(unknown)}; expected {list(SUITES)}")

    return [
        env_info(name)
        for s in suites
        for name in sorted(getattr(registry, s).ALL_ENVS)
    ]


def envs_table(suite: str | Iterable[str] | None = None) -> pd.DataFrame:
    """list_envs as a DataFrame, with the derived timing columns."""
    infos = list_envs(suite)
    df = pd.DataFrame([asdict(i) for i in infos])
    if df.empty:
        return df
    df["sim_steps_per_ctrl"] = [i.sim_steps_per_ctrl_step for i in infos]
    df["episode_s"] = [i.episode_seconds for i in infos]
    return df.set_index("name")


def _platform_dir(suite: str, platform: str) -> Path:
    root = Path(str(mjx_env.ROOT_PATH)) / suite
    return root if suite == "dm_control_suite" else root / platform


def _xmls_for(suite: str, platform: str) -> list[str]:
    """Task XMLs Playground ships for this platform.
    An empty list is meaningful, not a bug: some robots (barkour) carry no
    local XML and load their scene straight from Menagerie.
    """
    directory = _platform_dir(suite, platform)
    if not directory.exists():
        return []
    if suite == "dm_control_suite":
        return sorted(p.name for p in (directory / "xmls").glob(f"{platform}*.xml"))
    return sorted(p.name for p in directory.rglob("*.xml"))


def list_models(suite: str | Iterable[str] | None = None) -> list[ModelInfo]:
    """The robot / model families, each with the environments built on it."""
    by_platform: dict[tuple[str, str], list[str]] = {}
    for info in list_envs(suite):
        by_platform.setdefault((info.suite, info.platform), []).append(info.name)

    return [
        ModelInfo(
            platform=platform,
            suite=s,
            n_envs=len(names),
            envs=sorted(names),
            xmls=_xmls_for(s, platform),
        )
        for (s, platform), names in sorted(by_platform.items())
    ]


def models_table(
    suite: str | Iterable[str] | None = None, sep: str = ", "
) -> pd.DataFrame:
    """list_models as a DataFrame, with the env lists collapsed to strings.
    """
    models = list_models(suite)
    df = pd.DataFrame(
        [
            {
                "platform": m.platform,
                "suite": m.suite,
                "n_envs": m.n_envs,
                "n_xmls": len(m.xmls),
                "envs": sep.join(m.envs),
            }
            for m in models
        ]
    )
    return df.set_index("platform") if not df.empty else df


def show_models(suite: str | Iterable[str] | None = None) -> Any:
    """Model table for notebooks, one environment per line, nothing truncated.
    """
    df = models_table(suite, sep="\n")
    return (
        df.style.set_properties(
            subset=["envs"], **{"white-space": "pre-wrap", "text-align": "left"}
        )
        .set_properties(**{"vertical-align": "top"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}])
    )


def print_models(
    suite: str | Iterable[str] | None = None, show_xmls: bool = False
) -> None:
    """Plain-text listing of models and their environments.
    """
    current_suite = None
    for m in list_models(suite):
        if m.suite != current_suite:
            current_suite = m.suite
            print(f"\n{'=' * 60}\n{current_suite}\n{'=' * 60}")

        print(f"\n{m.platform}  ({m.n_envs} env{'s' if m.n_envs != 1 else ''}, "
              f"{len(m.xmls)} xml{'s' if len(m.xmls) != 1 else ''})")
        for name in m.envs:
            print(f"    - {name}")
        if show_xmls:
            for xml in m.xmls:
                print(f"      · {xml}")


def menagerie_available() -> bool:
    """Whether the MuJoCo Menagerie assets have been fetched yet.
    """
    return Path(str(mjx_env.MENAGERIE_PATH)).exists()


def ensure_menagerie() -> Path:
    """Download the Menagerie assets if missing. A few hundred MB, once."""
    mjx_env.ensure_menagerie_exists()
    return Path(str(mjx_env.MENAGERIE_PATH))


def describe_env(env_name: str, config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the environment and report its true dimensions.
    """
    env = registry.load(env_name, config_overrides=config_overrides)
    model = env.mj_model

    def _names(count: int, obj_type: int) -> list[str]:
        import mujoco

        found = [mujoco.mj_id2name(model, obj_type, i) for i in range(count)]
        return [n for n in found if n is not None]

    import mujoco

    info = env_info(env_name)
    return {
        "name": env_name,
        "suite": info.suite,
        "platform": info.platform,
        "task": info.task,
        "observation_size": env.observation_size,
        "action_size": env.action_size,
        "nq": model.nq, # number of generalized coordinates
        "nv": model.nv, # number of generalized velocities
        "nu": model.nu, # number of actuators
        "nbody": model.nbody, # number of bodies
        "ngeom": model.ngeom, # number of geoms, which are the collision shapes
        "nsensor": model.nsensor, # number of sensors
        "actuators": _names(model.nu, mujoco.mjtObj.mjOBJ_ACTUATOR),
        "sensors": _names(model.nsensor, mujoco.mjtObj.mjOBJ_SENSOR),
        "xml_path": str(getattr(env, "xml_path", "")),
    }
