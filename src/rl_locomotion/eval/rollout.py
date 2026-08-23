"""Roll a policy (or no policy) through an environment and keep the trajectory.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

Policy = Callable[[Any, jax.Array], jax.Array]

@dataclass
class Rollout:
    """One episode. Rollout is showing the trajectory of a policy through an environment,
    including states, actions, rewards, and metrics.
    """
    env_name: str
    states: list[Any]
    actions: np.ndarray
    rewards: np.ndarray
    ctrl_dt: float # this is the control timestep of the environment, not the simulation timestep!!
    metrics: list[dict[str, float]] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.rewards)

    @property
    def total_reward(self) -> float:
        return float(self.rewards.sum())

    @property
    def duration_s(self) -> float:
        return self.n_steps * self.ctrl_dt

    @property
    def terminated_early(self) -> bool:
        """True if the episode ended before the requested number of steps.
        For locomotion this usually means the robot fell over.
        """
        return bool(self.states[-1].done)

    def metrics_frame(self) -> Any:
        """Per-step metrics as a DataFrame, one column per term.
        Most locomotion tasks - Go1, T1, Spot joystick, Barkour and others, 11
        of 15 at the time of writing - clip the summed reward at zero:
        (reward = clip(sum(terms) * dt, 0.0, 10000.0)), so a badly behaved policy reports a flat zero return while the terms
        underneath are strongly negative. These columns are where that becomes
        visible. A few tasks (G1 and Apollo joystick among them) do not clip and
        can go negative, so check the task before reading anything into the
        sign of a return.
        """
        return pd.DataFrame(self.metrics)

    def summary(self) -> dict[str, Any]:
        return {
            "env": self.env_name,
            "steps": self.n_steps,
            "duration_s": round(self.duration_s, 2),
            "total_reward": round(self.total_reward, 2),
            "mean_reward": round(float(self.rewards.mean()), 4),
            "terminated_early": self.terminated_early,
        }


def random_policy(action_size: int) -> Policy:
    """Uniform noise in [-1, 1]. Useful as a baseline and for checking that an environment is wired up.
    """
    def rnd_policy(obs: Any, rng: jax.Array) -> jax.Array:
        del obs
        return jax.random.uniform(rng, (action_size,), minval=-1.0, maxval=1.0)

    return rnd_policy


def zero_policy(action_size: int) -> Policy:
    """Hold the default pose. The clearest look at a robot's starting state."""

    def zero_policy(obs: Any, rng: jax.Array) -> jax.Array:
        del obs, rng
        return jnp.zeros(action_size)

    return zero_policy


def rollout(
    env: Any,
    policy: Policy | None = None,
    n_steps: int = 100,
    seed: int = 0,
    stop_on_done: bool = True,
    env_name: str = "",
) -> Rollout:
    """Step env for n_steps control steps.
    `policy=None` means the zero action.
    """
    reset_fn = jax.jit(env.reset)
    step_fn = jax.jit(env.step)

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = reset_fn(reset_rng) 

    if policy is None:
        policy = zero_policy(env.action_size)

    states, actions, rewards, metrics = [state], [], [], []
    for _ in range(n_steps):
        rng, action_rng = jax.random.split(rng)
        action = policy(state.obs, action_rng)
        state = step_fn(state, action)

        states.append(state)
        actions.append(np.asarray(action))
        rewards.append(float(state.reward))
        metrics.append({k: float(v) for k, v in state.metrics.items()})

        if stop_on_done and bool(state.done):
            break

    return Rollout(
        env_name=env_name or type(env).__name__,
        states=states,
        actions=np.asarray(actions),
        rewards=np.asarray(rewards),
        ctrl_dt=float(env.dt),
        metrics=metrics,
    )
