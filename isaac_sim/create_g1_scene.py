#!/usr/bin/env python3
"""Create a minimal Isaac Sim scene USD with the G1 robot + ground plane.

Run once inside the Isaac Sim Python environment to generate a scene file
that dds_bridge.py can load without dynamic scene modifications.

Usage:
    ./python.sh /rdp/isaac_sim/create_g1_scene.py \
        --robot-usd /groot/gear_sonic/data/robots/g1/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd \
        --output /rdp/isaac_sim/g1_scene.usd
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Create G1 scene USD")
    parser.add_argument("--robot-usd", required=True, help="Path to robot USD asset")
    parser.add_argument("--output", default="/rdp/isaac_sim/g1_scene.usd", help="Output scene path")
    parser.add_argument("--height", type=float, default=0.85, help="Initial robot height (m)")
    args = parser.parse_args()

    try:
        from isaacsim import SimulationApp
    except ImportError:
        print("Error: run inside Isaac Sim Python environment.")
        sys.exit(1)

    sim_app = SimulationApp({"headless": True})

    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema
    import omni.usd

    stage = omni.usd.get_context().get_stage()

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Physics scene with gravity
    physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr(9.81)

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/GroundPlane")
    ground.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    ground.CreateNormalsAttr([(0, 0, 1)] * 4)
    collision = UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Reference the robot USD
    robot_xform = UsdGeom.Xform.Define(stage, "/World/G1")
    robot_xform.GetPrim().GetReferences().AddReference(args.robot_usd)
    translate_op = robot_xform.GetOrderedXformOps()
    if translate_op:
        translate_op[0].Set(Gf.Vec3d(0, 0, args.height))
    else:
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, args.height))

    stage.GetRootLayer().Export(args.output)
    print(f"Scene saved to: {args.output}")

    sim_app.close()

if __name__ == "__main__":
    main()
