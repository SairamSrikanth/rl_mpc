import argparse, functools, math, os, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description="Look at the warehouse: --gui opens the viewer (run without --headless), --preview writes a few fixed views.")
ap.add_argument("--gui", action="store_true")
ap.add_argument("--preview", action="store_true")
ap.add_argument("--dense", action="store_true")
ap.add_argument("--extra_north", type=float, default=8.0, help="extra ground behind the ramp foot")
ap.add_argument("--plain", action="store_true", help="show the collision proxies, no props")
ap.add_argument("--out", default="/tmp/wh_render")
ap.add_argument("--width", type=int, default=1280)
ap.add_argument("--height", type=int, default=720)
ap.add_argument("--fov", type=float, default=90.0)
ap.add_argument("--warm", type=int, default=24, help="render steps before each grab (textures and lights settle)")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import numpy as np
from isaaclab.sim import SimulationCfg, SimulationContext

STATIC_VIEWS = {
    "atrium_high": ((19.0, 11.5, 5.0), (7.0, 6.0, 1.5)), "atrium_stairs": ((13.5, 7.0, 1.2), (5.0, 7.0, 3.0)), "ground_aisle": ((-19.0, -7.5, 1.3), (-3.0, -7.5, 1.0)),
    "upper_floor": ((-19.0, 0.0, 4.3), (1.0, 4.0, 3.3)), "ramp": ((15.0, 12.0, 1.2), (15.0, 2.5, 2.5)), "side_stairs": ((13.0, 4.3, 1.6), (7.0, 7.0, 1.4)),
    "rail_corner": ((8.0, -2.0, 4.6), (5.0, 2.5, 3.2)), "forklift": ((-8.0, 4.0, 1.6), (-12.0, 8.0, 0.8))}


def look_at(eye, target):
    """Camera axes (OpenCV: x right, y down, z forward) as the columns of a rotation matrix."""
    f = np.asarray(target, float) - np.asarray(eye, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([1.0, 0, 0])
    return np.stack([r, np.cross(f, r), f], 1)


def build_world():
    from spot_mpc.worlds.warehouse_static.layout import WarehouseCfg
    from spot_mpc.worlds.warehouse_static.world import Warehouse
    cfg = WarehouseCfg()
    cfg.dense, cfg.extra_north = args.dense, args.extra_north
    world = Warehouse(cfg)
    world.spawn()
    if not args.plain:
        world.dress()
    return world, STATIC_VIEWS, ((19.0, 11.5, 5.0), (7.0, 6.0, 1.5))


def main():
    sim = SimulationContext(SimulationCfg(dt=0.0025, device="cuda:0"))
    world, views, home = build_world()
    sim.reset()
    scene = getattr(world, "scene", None)
    if args.gui:
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=list(home[0]), target=list(home[1]))
        print("[render] viewer open: fly with RMB + WASD, close the window to quit")
        t0 = time.time()
        while app.is_running():
            if scene is not None:
                scene.step(time.time() - t0)
            app.update()
        os._exit(0)

    import omni.replicator.core as rep
    import omni.usd
    from pxr import Gf, UsdGeom
    stage = omni.usd.get_context().get_stage()
    cam = UsdGeom.Camera.Define(stage, "/World/ViewCam")
    cam.GetFocalLengthAttr().Set(18.0 / math.tan(math.radians(args.fov) / 2))
    cam.GetHorizontalApertureAttr().Set(36.0)
    cam.GetVerticalApertureAttr().Set(36.0 * args.height / args.width)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05, 120.0))
    pose = UsdGeom.Xformable(cam).AddTransformOp()
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rep.create.render_product("/World/ViewCam", (args.width, args.height))])
    import imageio.v2 as imageio
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, (eye, target) in views.items():
        R = look_at(eye, target)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = R[:, 0], -R[:, 1], -R[:, 2], eye       # USD cameras look down -Z with +Y up
        pose.Set(Gf.Matrix4d(M.T.tolist()))
        for _ in range(args.warm * 2):
            rep.orchestrator.step(rt_subframes=1, delta_time=0.0)
        image = np.asarray(rgb.get_data())[..., :3]
        imageio.imwrite(out / f"{name}.jpg", image, quality=92)
        print(f"[render] {name}: mean {image.mean():.1f}")
    os._exit(0)


if __name__ == "__main__":
    main()
