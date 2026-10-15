import argparse
import math

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser(description="Video of the elevation map the MPC plans on, in the robot's frame, with the Raibert footholds on it. Input: the .npz from run_warehouse.py --dump_map.")
ap.add_argument("--data", required=True)
ap.add_argument("--chase", default=None, help="the chase-camera mp4 recorded in the same run (same --video_every): shown on the left")
ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=float, default=20.0, help="playback frame rate; the data is 40 / video_every frames per second, so 20 is real time at --video_every 2")
ap.add_argument("--ahead", type=float, default=2.4, help="map shown this far ahead of the robot [m]")
ap.add_argument("--behind", type=float, default=1.0)
ap.add_argument("--half_width", type=float, default=1.4)
ap.add_argument("--cell", type=float, default=0.05, help="resampled cell size [m]")
ap.add_argument("--size", default="960,540", help="size of each panel w,h")
ap.add_argument("--elev", type=float, default=26.0)
ap.add_argument("--azim", type=float, default=195.0, help="view from behind the robot: 180 is straight behind, larger moves to the robot's right")
ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
ap.add_argument("--first", type=int, default=0)
ap.add_argument("--last", type=int, default=None)
ap.add_argument("--speed_label", default="", help="text in the corner, e.g. '1x speed'")
args = ap.parse_args()

LEGS = ("FL", "FR", "HL", "HR")
HIP = np.array([(0.30, 0.11), (0.30, -0.11), (-0.30, 0.11), (-0.30, -0.11)])        # hip positions in the body frame (the legs are drawn from here)
BLUE, GREEN, RED, GOLD = "#1f77b4", "#2ca02c", "#d62728", "#e6a700"


def to_local(p, x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(p, float) - np.array([x, y, z])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


def sampled_map(d, i, xr, yr, zr, yaw, Xf, Yf):
    """The elevation window resampled onto a grid aligned with the robot (x forward, y left): z relative to the robot, nan where the robot has not seen the cell."""
    res, o = float(d["res"]), d["origin"][i]
    c, s = math.cos(yaw), math.sin(yaw)
    wx, wy = xr + c * Xf - s * Yf, yr + s * Xf + c * Yf
    ix, iy = np.floor((wx - o[0]) / res + 0.5).astype(int), np.floor((wy - o[1]) / res + 0.5).astype(int)
    g, seen = d["grid"][i], d["seen"][i]
    ok = (ix >= 0) & (ix < g.shape[1]) & (iy >= 0) & (iy < g.shape[0])
    ix, iy = np.clip(ix, 0, g.shape[1] - 1), np.clip(iy, 0, g.shape[0] - 1)
    h = g[iy, ix].astype(float)
    return np.where(ok & seen[iy, ix], h - zr, np.nan)


def body_box(zc=0.0, hl=0.42, hw=0.15, hh=0.09):
    v = np.array([(x, y, zc + z) for x in (-hl, hl) for y in (-hw, hw) for z in (-hh, hh)])
    f = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
    return [v[list(q)] for q in f]


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def caption(img, text, xy=(12, 10), size=26):
    d = ImageDraw.Draw(img, "RGBA")
    f = font(size)
    b = d.textbbox((0, 0), text, font=f)
    d.rounded_rectangle((xy[0] - 8, xy[1] - 6, xy[0] + b[2] + 8, xy[1] + b[3] + 10), radius=8, fill=(0, 0, 0, 170))
    d.text((xy[0] - b[0], xy[1] - b[1]), text, font=f, fill=(255, 255, 255, 255))


def main():
    d = np.load(args.data)
    n = len(d["t"])
    W, H = (int(v) for v in args.size.split(","))
    chase = imageio.get_reader(args.chase) if args.chase else None
    if chase is not None and abs(chase.count_frames() - n) > 2:
        print(f"[warn] the chase video has {chase.count_frames()} frames and the data {n}: they were not recorded with the same --video_every")
    cursor, current = -1, None                                  # the chase video is read in order, one frame at a time
    Xf, Yf = np.meshgrid(np.arange(-args.behind, args.ahead + 1e-9, args.cell), np.arange(-args.half_width, args.half_width + 1e-9, args.cell))
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    cmap = plt.get_cmap("turbo")
    handles = [Line2D([], [], marker="o", ls="", mfc=GREEN, mec="k", ms=9, label="foot on the ground"),
               Line2D([], [], marker="o", ls="", mfc=BLUE, mec="k", ms=9, label="foot in swing"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=GOLD, mew=2.2, ms=10, label="Raibert foothold (raw)"),
               Line2D([], [], marker="*", ls="", mfc=RED, mec="k", ms=14, label="touchdown target (edges avoided)")]
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8, macro_block_size=None)
    last = n if args.last is None else min(args.last, n)
    for i in range(args.first, last, args.every):
        x0 = d["x0"][i]
        yaw, xr, yr, zr = float(x0[2]), float(x0[3]), float(x0[4]), float(x0[5])
        Z = sampled_map(d, i, xr, yr, zr, yaw, Xf, Yf)
        ax.cla()
        col = cmap(np.clip((Z + 0.9) / 2.5, 0.0, 1.0))
        col[np.isnan(Z)] = (0, 0, 0, 0)
        ax.plot_surface(Xf, Yf, Z, facecolors=col, rstride=1, cstride=1, shade=True, linewidth=0, antialiased=False, zorder=1)
        ax.add_collection3d(Poly3DCollection(body_box(), facecolor=(1.0, 0.8, 0.0, 0.55), edgecolor="k", linewidths=0.8, zorder=3))
        feet = to_local(d["feet"][i], xr, yr, zr, yaw)
        td = to_local(d["td"][i], xr, yr, zr, yaw)
        raw = to_local(d["raw"][i], xr, yr, zr, yaw)
        con = d["contact"][i]
        for k in range(4):
            ax.plot([HIP[k, 0], feet[k, 0]], [HIP[k, 1], feet[k, 1]], [-0.05, feet[k, 2]], color="#444", lw=2.0, zorder=5)
            if con[k]:
                ax.scatter(*feet[k], s=70, c=GREEN, edgecolors="k", depthshade=False, zorder=9)
            else:
                ax.scatter(*feet[k], s=70, c=BLUE, edgecolors="k", depthshade=False, zorder=9)
                ax.plot([feet[k, 0], td[k, 0]], [feet[k, 1], td[k, 1]], [feet[k, 2], td[k, 2]], color=RED, lw=1.2, alpha=0.8, zorder=8)
                ax.plot([raw[k, 0], td[k, 0]], [raw[k, 1], td[k, 1]], [raw[k, 2], td[k, 2]], color=GOLD, lw=1.6, ls="--", zorder=8)
                ax.scatter(*raw[k], s=120, facecolors="none", edgecolors=GOLD, linewidths=2.2, depthshade=False, zorder=10)
                ax.scatter(*td[k], s=190, marker="*", c=RED, edgecolors="k", depthshade=False, zorder=11)
            ax.text(feet[k, 0], feet[k, 1], feet[k, 2] - 0.09, LEGS[k], fontsize=8, ha="center", zorder=12)
        ax.set_xlim(-args.behind, args.ahead)
        ax.set_ylim(-args.half_width, args.half_width)
        ax.set_zlim(-0.9, 1.6)
        ax.set_box_aspect((args.ahead + args.behind, 2 * args.half_width, 2.5), zoom=1.3)
        ax.view_init(elev=args.elev, azim=args.azim)
        ax.set_axis_off()
        ax.set_position([0.0, 0.04, 1.0, 0.98])
        fig.legend(handles=handles, loc="lower right", fontsize=9, frameon=True, framealpha=0.85)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.canvas.draw()
        right = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        caption(right, "Elevation map the MPC plans on (robot frame, colour = height)", size=20)
        caption(right, f"t = {d['t'][i]:5.1f} s", xy=(12, H - 52), size=22)
        if chase is not None:
            while cursor < i:
                try:
                    current, cursor = chase.get_next_data(), cursor + 1
                except (IndexError, StopIteration):
                    break
            left = Image.fromarray(current).resize((W, H))
            caption(left, "Isaac Sim, chase camera", size=20)
            if args.speed_label:
                caption(left, args.speed_label, xy=(12, H - 52), size=22)
            frame = Image.new("RGB", (2 * W, H))
            frame.paste(left, (0, 0))
            frame.paste(right, (W, 0))
        else:
            frame = right
        writer.append_data(np.asarray(frame))
        if (i - args.first) % 60 == 0:
            print(f"[render] frame {i}/{last}", flush=True)
    writer.close()
    print(f"[video] {args.out}")


if __name__ == "__main__":
    main()
