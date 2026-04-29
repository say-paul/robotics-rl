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

    from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Sdf, Gf, PhysxSchema
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
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Dome light for ambient illumination
    dome = UsdLux.DomeLight.Define(stage, "/DomeLight")
    dome.CreateIntensityAttr(300.0)

    # Distant light for directional shadows
    dist_light = UsdLux.DistantLight.Define(stage, "/DistantLight")
    dist_light.CreateIntensityAttr(500.0)
    dist_xform = UsdGeom.Xformable(dist_light.GetPrim())
    dist_xform.AddRotateXYZOp().Set(Gf.Vec3f(-45, 30, 0))

    # Camera — captured from user's Isaac Sim viewport
    camera = UsdGeom.Camera.Define(stage, "/RobotCamera")
    cam_xform = UsdGeom.Xformable(camera.GetPrim())
    cam_mat = Gf.Matrix4d(
        -0.699875771292428, 0.7142645901609784, 0.0, 0.0,
        -0.31820982090342315, -0.3117995024608676, 0.8952784930655752, 0.0,
        0.6394657259294214, 0.626583725855792, 0.44550692458617625, 0.0,
        5.029047112805129, 4.372604376025868, 3.713891269049331, 1.0,
    )
    cam_xform.AddTransformOp().Set(cam_mat)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))

    # Reference the robot USD
    robot_xform = UsdGeom.Xform.Define(stage, "/World/G1")
    robot_xform.GetPrim().GetReferences().AddReference(args.robot_usd)
    translate_op = robot_xform.GetOrderedXformOps()
    if translate_op:
        translate_op[0].Set(Gf.Vec3d(0, 0, args.height))
    else:
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(0, 0, args.height))

    # Keep the USD's original stiffness (strong enough to hold against gravity)
    # but set damping to prevent overshoot. Ratio 0.025 from Unitree's config.
    DAMPING_RATIO = 0.025
    for prim in stage.Traverse():
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive and drive.GetStiffnessAttr().HasValue():
            kp = drive.GetStiffnessAttr().Get()
            if kp and kp > 0:
                drive.GetDampingAttr().Set(kp * DAMPING_RATIO)

    stage.GetRootLayer().Export(args.output)
    print(f"Scene saved to: {args.output}")

    sim_app.close()

if __name__ == "__main__":
    main()
