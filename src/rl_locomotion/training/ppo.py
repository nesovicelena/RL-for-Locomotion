"""PPO training loop (Brax) for the Go1 ERFI study.

One call to `train()` trains one (condition, seed) pair and leaves a
self-describing run directory behind:

    run_dir/
      spec.json          what was asked for
      env_config.json    the full environment ConfigDict
      ppo_config.json    the full PPO ConfigDict (network sizes included)
      curve.json         eval reward after every eval
      params_<step>      brax checkpoints, one per eval
      params_latest      the most recent of those
      params_final       written when training finishes
      summary.json       wall time and final reward

`load_policy()` rebuilds the network from those files, so evaluation never
depends on the training process still being alive.
"""
from __future__ import annotations

import functools
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax

# brax 0.14.2 still calls jax.device_put_replicated, which JAX 0.10 removed from
# the public namespace. The implementation still exists; re-export it before
# brax is imported.
if not hasattr(jax, "device_put_replicated"):  # pragma: no cover - version dependent
    from jax._src.api import device_put_replicated as _dpr

    jax.device_put_replicated = _dpr

from brax.io import model as brax_model  # noqa: E402
from brax.training.acme import running_statistics  # noqa: E402
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402
from ml_collections import config_dict  # noqa: E402
from mujoco_playground import wrapper  # noqa: E402
from mujoco_playground.config import locomotion_params  # noqa: E402

from rl_locomotion.envs import erfi  # noqa: E402

PolicyFn = Callable[[Any, jax.Array], tuple[jax.Array, Any]]


@dataclass
class TrainSpec:
    """Everything that distinguishes one run from another."""

    condition: str  # one of erfi.CONDITIONS
    seed: int = 0
    # Robot ("go1" or "a1") and terrain ("flat_terrain" or "rough_terrain").
    # Both are stored in env_config.json so evaluation rebuilds the same model
    # on the same scene.
    robot: str = "go1"
    task: str = "flat_terrain"
    num_timesteps: int = 200_000_000
    num_evals: int = 10
    # Observation / network. Defaults follow the paper's blind A1 setup:
    # 7-step history (192-dim state) and a [512, 512] policy.
    history_len: int = 7
    policy_layers: tuple[int, ...] = (512, 512)
    value_layers: tuple[int, ...] = (512, 256, 128)
    # The paper's critic sees only the policy state. Playground's default is an
    # asymmetric critic on the privileged state, which trains more reliably;
    # the deployed policy is identical either way.
    symmetric_critic: bool = False
    # ERFI torque limits (Nm) and MJX backend.
    rfi_lim: float = 7.0
    rao_lim: float = 7.0
    impl: str = "warp"
    # Free-form overrides applied last (dotted keys allowed, e.g. "noise_config.level").
    env_overrides: dict[str, Any] = field(default_factory=dict)
    ppo_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.condition not in erfi.CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}; choose from {list(erfi.CONDITIONS)}")
        if self.task not in erfi.TASKS:
            raise ValueError(f"unknown task {self.task!r}; choose from {list(erfi.TASKS)}")
        if self.robot not in erfi.ROBOTS:
            raise ValueError(f"unknown robot {self.robot!r}; choose from {list(erfi.ROBOTS)}")
        self.policy_layers = tuple(self.policy_layers)
        self.value_layers = tuple(self.value_layers)


def _set_dotted(cfg: config_dict.ConfigDict, key: str, value: Any) -> None:
    node = cfg
    *parents, leaf = key.split(".")
    for p in parents:
        node = node[p]
    node[leaf] = value


def env_config(spec: TrainSpec) -> config_dict.ConfigDict:
    cfg = erfi.condition_config(
        spec.condition, robot=spec.robot, task=spec.task, impl=spec.impl, history_len=spec.history_len
    )
    cfg.erfi.rfi_lim = spec.rfi_lim
    cfg.erfi.rao_lim = spec.rao_lim
    for k, v in spec.env_overrides.items():
        _set_dotted(cfg, k, v)
    return cfg


def ppo_config(spec: TrainSpec) -> config_dict.ConfigDict:
    """Playground's tuned Go1 PPO settings with the study's network choices."""
    params = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")
    params.num_timesteps = spec.num_timesteps
    params.num_evals = spec.num_evals
    params.network_factory.policy_hidden_layer_sizes = spec.policy_layers
    params.network_factory.value_hidden_layer_sizes = spec.value_layers
    params.network_factory.value_obs_key = "state" if spec.symmetric_critic else "privileged_state"
    for k, v in spec.ppo_overrides.items():
        _set_dotted(params, k, v)
    return params


def _dump(path: Path, obj: Any) -> None:
    if isinstance(obj, config_dict.ConfigDict):
        obj = obj.to_dict()
    path.write_text(json.dumps(obj, indent=2, default=str))


def _network_factory(params: config_dict.ConfigDict):
    nf = params.network_factory.to_dict()
    nf["policy_hidden_layer_sizes"] = tuple(nf["policy_hidden_layer_sizes"])
    nf["value_hidden_layer_sizes"] = tuple(nf["value_hidden_layer_sizes"])
    return functools.partial(ppo_networks.make_ppo_networks, **nf)


def train(
    spec: TrainSpec,
    run_dir: Path | str,
    progress_fn: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Train one run. Returns the brax params, the inference-fn builder and the curve."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = env_config(spec)
    params = ppo_config(spec)
    _dump(run_dir / "spec.json", asdict(spec))
    _dump(run_dir / "env_config.json", env_cfg)
    _dump(run_dir / "ppo_config.json", params)

    env = erfi.load(env_cfg)
    eval_env = erfi.load(env_cfg)
    randomizer = (
        erfi.domain_randomizer(spec.task, spec.robot) if erfi.uses_domain_randomization(spec.condition) else None
    )

    curve: list[dict[str, float]] = []
    t0 = time.time()

    def _progress(num_steps: int, metrics: dict[str, Any]) -> None:
        row = {
            "step": int(num_steps),
            "reward": float(metrics["eval/episode_reward"]),
            "reward_std": float(metrics["eval/episode_reward_std"]),
            "wall_s": time.time() - t0,
        }
        curve.append(row)
        _dump(run_dir / "curve.json", curve)
        print(
            f"[{spec.condition} seed={spec.seed}] step {row['step']:>12,d}  "
            f"reward {row['reward']:8.3f} ± {row['reward_std']:.3f}  {row['wall_s']:7.0f}s",
            flush=True,
        )
        if progress_fn is not None:
            progress_fn(num_steps, metrics)

    def _checkpoint(step: int, make_policy: Any, ps: Any) -> None:
        del make_policy
        brax_model.save_params(str(run_dir / f"params_{step:012d}"), ps)
        brax_model.save_params(str(run_dir / "params_latest"), ps)

    training_params = dict(params)
    del training_params["network_factory"]

    make_inference_fn, ps, metrics = ppo.train(
        **training_params,
        network_factory=_network_factory(params),
        randomization_fn=randomizer,
        progress_fn=_progress,
        policy_params_fn=_checkpoint,
        seed=spec.seed,
        environment=env,
        eval_env=eval_env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )

    brax_model.save_params(str(run_dir / "params_final"), ps)
    summary = {
        "condition": spec.condition,
        "seed": spec.seed,
        "robot": spec.robot,
        "task": spec.task,
        "wall_s": time.time() - t0,
        "final_reward": curve[-1]["reward"] if curve else None,
        "num_timesteps": spec.num_timesteps,
    }
    _dump(run_dir / "summary.json", summary)
    return {"params": ps, "make_inference_fn": make_inference_fn, "curve": curve, "summary": summary}


def is_finished(run_dir: Path | str) -> bool:
    return (Path(run_dir) / "params_final").exists()


def load_env_config(run_dir: Path | str, **overrides: Any) -> config_dict.ConfigDict:
    """The env config a run was trained with, as a ConfigDict, with optional overrides."""
    cfg = config_dict.ConfigDict(json.loads((Path(run_dir) / "env_config.json").read_text()))
    # Runs from before robot/terrain became part of the config were all Go1 on flat ground.
    if "task" not in cfg:
        cfg.task = "flat_terrain"
    if "robot" not in cfg:
        cfg.robot = "go1"
    for k, v in overrides.items():
        _set_dotted(cfg, k, v)
    return cfg


def load_policy(
    run_dir: Path | str,
    env: Any,
    checkpoint: str = "params_final",
    deterministic: bool = True,
) -> PolicyFn:
    """Rebuild a trained policy from a run directory. `env` decides the obs shapes."""
    run_dir = Path(run_dir)
    params_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    nf = dict(params_cfg["network_factory"])
    nf["policy_hidden_layer_sizes"] = tuple(nf["policy_hidden_layer_sizes"])
    nf["value_hidden_layer_sizes"] = tuple(nf["value_hidden_layer_sizes"])

    obs_size = env.observation_size
    if isinstance(obs_size, dict):
        obs_shape = {k: (v,) if isinstance(v, int) else tuple(v) for k, v in obs_size.items()}
    else:
        obs_shape = (obs_size,)

    preprocess = running_statistics.normalize if params_cfg.get("normalize_observations", True) else (lambda x, y: x)
    network = ppo_networks.make_ppo_networks(
        obs_shape, env.action_size, preprocess_observations_fn=preprocess, **nf
    )
    ps = brax_model.load_params(str(run_dir / checkpoint))
    return ppo_networks.make_inference_fn(network)(ps, deterministic=deterministic)
