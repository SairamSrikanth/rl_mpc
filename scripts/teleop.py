import argparse, functools, math, os, sys
from pathlib import Path
from isaaclab.app import AppLauncher

print = functools.partial(print, flush=True)
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description="Drive the MPC-controlled Spot with the keyboard: W/S or up/down walk, A/D or left/right step sideways, Q/E turn. "
                                         "Run without --headless; click the viewer window so it has keyboard focus.")
ap.add_argument("--world", default="warehouse", choices=["warehouse", "test"], help="the two-floor warehouse or the small test world (flat, stairs, ramp, rubble)")
ap.add_argument("--spawn", default="12,7,0,3.1416", help="warehouse start as x,y,floor_z,yaw (the default faces the staircase; the upper floor is z=3)")
ap.add_argument("--speed", type=float, default=0.45, help="forward / backward speed [m/s] (the controller is tuned for 0.45)")
ap.add_argument("--strafe", type=float, default=0.25, help="sideways speed [m/s]")
ap.add_argument("--yaw_rate", type=float, default=0.6, help="turn rate [rad/s]")
ap.add_argument("--dense", action="store_true", help="(the default now) racks, pallets, box piles, barrels and forklifts on both floors")
ap.add_argument("--empty", action="store_true", help="the bare building instead: no racks, pallets, forklifts or clutter")
ap.add_argument("--no_visual", action="store_true", help="skip the NVIDIA props and guard rails (faster to start)")
ap.add_argument("--follow", action="store_true", help="start with the camera following the robot (the C key chases); by default the camera is yours to move with the mouse")
ap.add_argument("--print_keys", action="store_true", help="print the name of every key the input system reports (for a keyboard whose keys do not respond)")
ap.add_argument("--demo", action="store_true", help="no keyboard: replay a fixed key sequence (forward, sideways, turns), for a headless check")
ap.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE", help="config overrides")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
args.visual = not args.no_visual and not args.empty and not getattr(args, "headless", False)
if args.visual:
    args.enable_cameras = True
if getattr(args, "headless", False) and not args.demo:
    ap.error("the keyboard needs a window: run without --headless (or use --demo for a scripted headless run)")
app = AppLauncher(args).app
sys.path.insert(0, str(ROOT))
import torch
from spot_mpc.cli import apply_overrides
from spot_mpc.tasks.task import SpotTask, SpotTaskCfg
from spot_mpc.teleop import KeyCommand, key_name

LEGEND = """
  W / up      forward          Q   turn left
  S / down    backward         E   turn right
  A / left    step left        X or space   stop
  D / right   step right       R   reset the robot to its start
  + (or =)    faster           -   slower     (six speed steps, 0.4x ... 1.5x of --speed)
  C           chase camera: jump behind the robot and follow it; moving the camera yourself lets go, C chases again   [F is the viewport's frame-everything key: avoid it]

  Camera (Isaac Sim viewport): hold Alt + left mouse to orbit, middle mouse to pan, scroll to zoom.
  Do not hold the right mouse button while driving: it flies the camera with W A S D, the same keys as the robot.
"""
DEMO = [("W", 4.0), ("D", 3.0), ("E", 2.5), ("W", 3.0), ("Q", 2.5), ("S", 3.0), ("", 1.0)]       # (key held, seconds)


def build():
    cfg = SpotTaskCfg()
    cfg.scene.num_envs = 1
    cfg.terrain_source, cfg.rear_depth = "sensed", True                    # the sensed map the controller plans on; the rear camera lets it walk backward on stairs
    cfg.episode_length_s = 24 * 3600.0
    cfg.map_half_x, cfg.map_half_y = 4.0, 3.0
    spawn = None
    if args.world == "warehouse":
        cfg.world_type = "warehouse"
        cfg.warehouse.dense, cfg.warehouse.visual = not args.empty, args.visual
        spawn = [float(v) for v in args.spawn.split(",")]
    apply_overrides(cfg, args.set)
    env = SpotTask(cfg)
    env.fixed_spawn = True                                                  # keep our spawn pose and command: no training lanes here
    if spawn is not None:
        env.spawn_pose = torch.tensor([[spawn[0], spawn[1], spawn[3]]], device=env.device)
        env.spawn_floor_z = torch.full((1,), spawn[2], device=env.device)
    with torch.inference_mode():
        env.reset()
    if not getattr(args, "headless", False):                                # a starting view behind and above the robot; after that the camera is the user's
        x = env.x0[0]
        yaw = float(x[2])
        eye = (float(x[3]) - 3.0 * math.cos(yaw), float(x[4]) - 3.0 * math.sin(yaw), float(x[5]) + 1.6)
        env.sim.set_camera_view(eye, (float(x[3]), float(x[4]), float(x[5]) - 0.1))
    return env


class CameraFollow:
    """C puts the camera in a chase view (behind and above the robot, looking at it) and carries it along with the robot. The moment you move the camera yourself (orbit, pan, zoom,
    fly) the script lets go and the camera stays where you put it; C snaps back to the chase view. The script notices by comparing the camera with where it last left it."""
    BEHIND, ABOVE, LOOK, TOL = 3.0, 1.6, 0.1, 2e-3

    def __init__(self, sim):
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf
        self.sim, self.state, self.Gf = sim, ViewportCameraState(), Gf
        self.on, self.mine, self.warned = False, None, False

    def position(self):
        p = self.state.position_world
        return [float(p[0]), float(p[1]), float(p[2])]

    def target(self):
        p = self.state.target_world
        return [float(p[0]), float(p[1]), float(p[2])]

    def _pose(self):
        return self.position() + self.target()

    def chase(self, pos, yaw):
        eye = (pos[0] - self.BEHIND * math.cos(yaw), pos[1] - self.BEHIND * math.sin(yaw), pos[2] + self.ABOVE)
        self.sim.set_camera_view(eye, (pos[0], pos[1], pos[2] - self.LOOK))
        self.mine = self._pose()

    def press(self, pos, yaw):
        self.on = True
        self.chase(pos, yaw)
        print("[teleop] chase camera (it lets go as soon as you move the camera; C chases again)")

    def update(self, pos):
        if not self.on:
            return
        if max(abs(a - b) for a, b in zip(self._pose(), self.mine)) > self.TOL:        # someone else moved the camera
            self.on = False
            print("[teleop] camera released: it is yours (C to chase again)")
            return
        want = [pos[0], pos[1], pos[2] - self.LOOK]
        if max(abs(a - b) for a, b in zip(self.target(), want)) < 1e-3:
            return
        try:
            self.state.set_target_world(self.Gf.Vec3d(*want), False)         # camera and orbit centre both move by the robot's step
            self.mine = self._pose()
        except Exception as ex:
            if not self.warned:
                print(f"[teleop] chase camera is not available here: {ex}")
                self.warned = True
            self.on = False


def main():
    env = build()
    keys = KeyCommand(args.speed, args.strafe, args.yaw_rate)
    zero = torch.zeros(1, env.cfg.action_space, device=env.device)
    flags = {"reset": False, "follow": False}
    cam = None if getattr(args, "headless", False) else CameraFollow(env.sim)
    if cam is not None and args.follow:
        flags["follow"] = True
    sub = None
    if not args.demo:
        import carb
        import omni.appwindow
        inp = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()

        def on_key(event, *_):
            name = key_name(event.input)
            if args.print_keys and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                print(f"[teleop] key {name}")
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if name in ("X", "SPACE"):
                    keys.clear()
                elif name == "R":
                    flags["reset"] = True
                elif name == "C":
                    flags["follow"] = True
                else:
                    before = keys.level
                    keys.press(name)
                    if keys.level != before:
                        print(f"[teleop] speed x{keys.scale:.1f}: forward {keys.scale * args.speed:.2f} m/s, sideways {keys.scale * args.strafe:.2f} m/s")
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                keys.release(name)
            return True

        sub = inp.subscribe_to_keyboard_events(keyboard, on_key)
        print(LEGEND)
        print("[teleop] click the viewer window first; the robot stands for a second, then trots.")
    demo, t_demo, step = list(DEMO), 0.0, 0
    while app.is_running():
        if args.demo:
            if not demo:
                break
            key, dur = demo[0]
            keys.clear()
            if key:
                keys.press(key)
            t_demo += env.step_dt
            if t_demo >= dur:
                demo.pop(0)
                t_demo = 0.0
        if flags["reset"]:
            flags["reset"] = False
            with torch.inference_mode():
                env.reset()
            keys.clear()
            print("[teleop] reset")
        if flags["follow"] and cam is not None:
            flags["follow"] = False
            cam.press([float(v) for v in env.x0[0, 3:6]], float(env.x0[0, 2]))
        vx, vy, wz = keys.command()
        env.cmd_target[:] = torch.tensor([vx, vy, wz], device=env.device)
        with torch.inference_mode():
            _, _, term, _, _ = env.step(zero)
        step += 1
        if bool(term[0]):
            print("[teleop] fell: reset to the start")
        if cam is not None:
            cam.update([float(v) for v in env.x0[0, 3:6]])
        if step % int(2.0 / env.step_dt) == 0:
            x = env.x0[0]
            if cam is not None and cam.on and args.demo:
                print(f"[teleop] camera at {[round(v, 2) for v in cam.position()]}")
            print(f"[teleop] pos ({float(x[3]):6.2f}, {float(x[4]):6.2f}, {float(x[5]):5.2f})  yaw {math.degrees(float(x[2])):7.1f} deg  command vx {vx:+.2f} vy {vy:+.2f} wz {wz:+.2f}")
    if sub is not None:
        inp.unsubscribe_to_keyboard_events(keyboard, sub)
    os._exit(0)


if __name__ == "__main__":
    main()
