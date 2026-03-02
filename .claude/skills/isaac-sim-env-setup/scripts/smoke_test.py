"""Smoke test: load Unitree Go2 in Isaac Sim headlessly.

Confirms the physics engine can load a robot model, create
an articulation, and step the simulation without errors.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh /tmp/smoke_test.py --headless --enable_cameras
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

print("=== Smoke Test: Load Go2 ===")

assets_root = get_assets_root_path()
if assets_root is None:
    print("FAIL: Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit(1)

world = World(stage_units_in_meters=1.0, physics_dt=1/200.0, rendering_dt=1/60.0)
world.scene.add_default_ground_plane()

go2_path = assets_root + "/Isaac/Robots/Unitree/Go2/go2.usd"
print(f"Loading: {go2_path}")
add_reference_to_stage(usd_path=go2_path, prim_path="/World/Go2")

world.reset()

for i in range(30):
    world.step(render=True)

print("Smoke test PASSED: Go2 loaded and simulated 30 steps.")
simulation_app.close()
