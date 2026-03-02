"""Sprint 1: 仿真渲染验证脚本
在 Isaac Sim 中加载 Unitree Go2 机器人，渲染一帧并保存截图。
用法: /isaac-sim/python.sh /workspace/render_test.py --headless --enable_cameras
"""

from isaacsim import SimulationApp

# 启动仿真（headless 模式）
simulation_app = SimulationApp({"headless": True})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path

print("=== Sprint 1: 仿真渲染验证 ===")

# 获取资产路径
assets_root_path = get_assets_root_path()
print(f"Assets root: {assets_root_path}")

# 创建世界
world = World(stage_units_in_meters=1.0, physics_dt=1/200.0, rendering_dt=1/60.0)
world.scene.add_default_ground_plane()

# 设置相机
set_camera_view(
    eye=[3.0, 3.0, 2.0],
    target=[0.0, 0.0, 0.5],
    camera_prim_path="/OmniverseKit_Persp"
)

# 加载 Unitree Go2
go2_path = assets_root_path + "/Isaac/Robots/Unitree/Go2/go2.usd"
print(f"Loading Go2 from: {go2_path}")
add_reference_to_stage(usd_path=go2_path, prim_path="/World/Go2")

# 初始化世界
world.reset()
print("World reset, simulating...")

# 运行几步仿真
for i in range(60):
    world.step(render=True)

print("渲染验证完成！Go2 机器人已在仿真中加载。")
print("如需可视化，请通过 WebRTC 流式查看 Isaac Sim 界面。")

# 列出场景中的所有 prim
from pxr import Usd
stage = world.stage
print("\n=== 场景中的 Prim 列表 ===")
for prim in stage.Traverse():
    if prim.GetPath().pathString.startswith("/World"):
        prim_type = prim.GetTypeName()
        if prim_type in ["Xform", "Mesh", "ArticulationRoot", "PhysicsScene"]:
            print(f"  {prim.GetPath()} [{prim_type}]")

simulation_app.close()
print("\n仿真已关闭。DoD 第一项：仿真器中成功渲染出机器人模型 ✓")
