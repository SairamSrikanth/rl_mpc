from types import SimpleNamespace

import numpy as np

from .control.footholds import foothold
from .control.gait import command_world


class MapRecorder:
    """Logs what robot `i` senses and plans, tick by tick, for the map video: the elevation window it plans on, its feet, the
    touchdown targets (after the safe-foothold shift) and the raw Raibert points (before it)."""

    def __init__(self, env, i=0):
        self.env, self.i, self.rows = env, i, []

    def raw_raibert(self):
        env, i = self.env, self.i
        g, x0 = env.gait, env.xe
        v_w, wz = command_world(x0, env.cmd, env.anchor[0])
        td = g.time_to_touchdown()
        com_td = x0[:, None, 3:5] + v_w[:, None, :] * td[..., None]
        yaw_td = x0[:, 2:3] + wz[:, None] * td
        as_is = SimpleNamespace(project=lambda xy, margin: xy, height=env.terrain.height)         # the Raibert point with no edge avoidance
        return foothold(env.P, as_is, com_td, yaw_td, x0[:, None, 9:11], v_w[:, None, :], g.stance_duration[:, None], env.foot_res)[i]

    def capture(self, t):
        env, i, em = self.env, self.i, self.env.elev_map
        self.rows.append(dict(
            t=t, x0=env.x0[i].cpu().numpy(), feet=env.legs.foot_pos_w()[i].cpu().numpy(), contact=env.gait.contact()[i].cpu().numpy(),
            td=env.swing.p_td[i].cpu().numpy(), raw=self.raw_raibert().cpu().numpy(), swing=env.gait.swing_progress()[i].cpu().numpy(),
            cmd=env.cmd[i].cpu().numpy(), wup=float(env._wup[i]), wback=float(env._wback[i]),
            origin=em.origin[i].cpu().numpy(), grid=em.grid[i].half().cpu().numpy(), seen=em.seen[i].cpu().numpy()))

    def save(self, path):
        r = self.rows
        np.savez_compressed(path, res=self.env.elev_map.res, foot_z_offset=self.env.cfg.foot_z_offset, **{k: np.stack([row[k] for row in r]) for k in r[0]})
