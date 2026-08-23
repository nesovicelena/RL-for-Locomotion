"""Open a Playground environment in the interactive MuJoCo viewer.
This works for Mac, Linux, and Windows. On macOS, the viewer must be launched with `mjpython`
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "env_name", nargs="?", help="e.g. Go1JoystickFlatTerrain"
    )
    parser.add_argument(
        "--list", action="store_true", help="list environment names and exit"
    )
    args = parser.parse_args()

    import mujoco
    import mujoco.viewer
    from mujoco_playground import registry as pg_registry

    from rl_locomotion.envs.registry import print_models

    if args.list:
        print_models()
        return 0

    if args.env_name is None:
        parser.error("env_name is required (or pass --list)")

    if args.env_name not in pg_registry.ALL_ENVS:
        print(f"unknown environment {args.env_name!r}", file=sys.stderr)
        print("run with --list to see the options", file=sys.stderr)
        return 1

    # The macOS viewer must run on the main thread of a special launcher.
    if sys.platform == "darwin" and mujoco.viewer._MJPYTHON is None:
        print(
            "On macOS the viewer must be launched with mjpython:\n"
            f"    mjpython scripts/view_model.py {args.env_name}",
            file=sys.stderr,
        )
        return 1

    print(f"loading {args.env_name} ...")
    env = pg_registry.load(args.env_name)
    model = env.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def key_callback(keycode: int) -> None:
        if chr(keycode).lower() != "p":
            return
        cam = viewer.cam
        print(
            "\nhero_shot(env,"
            f" azimuth={cam.azimuth:.1f},"
            f" elevation={cam.elevation:.1f},"
            f" distance={cam.distance:.3f},"
            f" lookat=({cam.lookat[0]:.3f}, {cam.lookat[1]:.3f}, {cam.lookat[2]:.3f}))"
        )

    print("viewer open — press P to print camera parameters, Esc to quit")
    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            # The model is held in its reset pose on purpose: this tool is for
            # composing figures, not for watching the physics!
            viewer.sync()
            time.sleep(1 / 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
