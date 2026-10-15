import math

import numpy as np

ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"
KIT = ROOT + "/Isaac/Environments/Simple_Warehouse/Props/"
FORKLIFT = ROOT + "/Isaac/Props/Forklift/forklift.usd"
YELLOW = (1.0, 0.78, 0.0)

# (usd name, bounding box) of the cartons and crates used to fill racks and pallets
BOXES = [("SM_CardBoxA_01", (0.70, 0.50, 0.50)), ("SM_CardBoxB_01", (0.50, 0.50, 0.50)), ("SM_CardBoxC_01", (0.50, 0.50, 0.25)), ("SM_CardBoxD_01", (0.38, 0.25, 0.15)),
         ("SM_CratePlastic_A_01", (0.62, 0.50, 0.44)), ("SM_CratePlastic_B_01", (0.62, 0.50, 0.44)), ("SM_CratePlastic_E_01", (0.39, 0.57, 0.38)), ("SM_CardBoxA_02", (0.70, 0.50, 0.50))]
BARRELS = ["SM_BarelPlastic_A_01", "SM_BarelPlastic_A_02", "SM_BarelPlastic_B_01"]


class UsdBuilder:
    """Adds visual-only geometry and asset references under one root prim of the open stage."""

    def __init__(self, root):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdShade
        self.Gf, self.Usd, self.UsdGeom, self.UsdShade = Gf, Usd, UsdGeom, UsdShade
        self.stage = omni.usd.get_context().get_stage()
        self.root = root
        UsdGeom.Xform.Define(self.stage, root)
        self.n = 0
        self.mats = {}

    def group(self, name):
        path = f"{self.root}/{name}"
        self.UsdGeom.Xform.Define(self.stage, path)
        return path

    def hide(self, path):
        prim = self.stage.GetPrimAtPath(path)
        if prim.IsValid():
            self.UsdGeom.Imageable(prim).MakeInvisible()

    def ref(self, parent, url, pos, yaw=0.0, scale=(1, 1, 1), name="r", flip=False, wrap=False):
        """Reference an asset at pos / yaw / scale. wrap=True puts the reference on a child prim (robot assets carry their own transform ops)."""
        self.n += 1
        path = f"{parent}/{name}_{self.n}"
        prim = self.stage.DefinePrim(path, "Xform")
        target = self.stage.DefinePrim(path + "/asset", "Xform") if wrap else prim
        target.GetReferences().AddReference(url)
        x = self.UsdGeom.Xformable(prim)
        x.AddTranslateOp().Set(self.Gf.Vec3d(*map(float, pos)))
        x.AddRotateZOp().Set(float(math.degrees(yaw)))
        if flip:
            x.AddRotateXOp().Set(180.0)
        x.AddScaleOp().Set(self.Gf.Vec3f(*map(float, scale)))
        return prim

    def material(self, color, rough=0.6, metal=0.0, opacity=1.0, emissive=None):
        import isaaclab.sim as sim_utils
        key = (tuple(round(c, 3) for c in color), rough, metal, opacity, emissive)
        if key not in self.mats:
            path = f"/World/Looks/Built_{len(self.mats)}"
            kw = dict(diffuse_color=tuple(color), roughness=rough, metallic=metal, opacity=opacity)
            if emissive is not None:
                kw["emissive_color"] = emissive
            cfg = sim_utils.PreviewSurfaceCfg(**kw)
            cfg.func(path, cfg)
            self.mats[key] = path
        return self.mats[key]

    def bind(self, prim, color, **kw):
        mat = self.UsdShade.Material(self.stage.GetPrimAtPath(self.material(color, **kw)))
        self.UsdShade.MaterialBindingAPI(prim).Bind(mat, self.UsdShade.Tokens.strongerThanDescendants)

    def box(self, parent, center, size, color, quat=None, name="b", **kw):
        """A unit cube scaled to `size`. Returns the translate and scale ops so a caller can animate it."""
        self.n += 1
        cube = self.UsdGeom.Cube.Define(self.stage, f"{parent}/{name}_{self.n}")
        cube.GetSizeAttr().Set(1.0)
        x = self.UsdGeom.Xformable(cube)
        t = x.AddTranslateOp()
        t.Set(self.Gf.Vec3d(*map(float, center)))
        if quat is not None:
            x.AddOrientOp().Set(self.Gf.Quatf(float(quat[0]), self.Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
        s = x.AddScaleOp()
        s.Set(self.Gf.Vec3f(*map(float, size)))
        self.bind(cube.GetPrim(), color, **kw)
        return t, s

    def cylinder(self, parent, center, radius, height, color, name="c", **kw):
        self.n += 1
        cyl = self.UsdGeom.Cylinder.Define(self.stage, f"{parent}/{name}_{self.n}")
        cyl.GetRadiusAttr().Set(float(radius))
        cyl.GetHeightAttr().Set(float(height))
        cyl.GetAxisAttr().Set("Z")
        self.UsdGeom.Xformable(cyl).AddTranslateOp().Set(self.Gf.Vec3d(*map(float, center)))
        self.bind(cyl.GetPrim(), color, **kw)

    def bar(self, parent, a, b, thick, color, name="bar", **kw):
        """Square bar of cross-section thick x thick from point a to point b."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        length = float(np.linalg.norm(d))
        q = self.Gf.Rotation(self.Gf.Vec3d(1, 0, 0), self.Gf.Vec3d(*(d / length))).GetQuat()
        im = q.GetImaginary()
        return self.box(parent, (a + b) / 2, (length, thick, thick), color, (q.GetReal(), im[0], im[1], im[2]), name, **kw)

    def strip_physics(self, root=None, uninstance=True):
        """Remove every physics schema and deactivate joints on imported assets: they are scenery.
        Instanced assets can only be edited after un-instancing them; with uninstance=False they are skipped."""
        from pxr import UsdPhysics
        root = self.stage.GetPrimAtPath(root or self.root)
        if uninstance:
            for _ in range(4):
                instances = [q for q in self.Usd.PrimRange(root) if q.IsInstance()]
                if not instances:
                    break
                for q in instances:
                    q.SetInstanceable(False)
        removed = 0
        apis = (UsdPhysics.CollisionAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.MeshCollisionAPI)
        for q in self.Usd.PrimRange(root, self.Usd.TraverseInstanceProxies()):
            if q.IsInstanceProxy():
                continue
            if q.IsA(UsdPhysics.Joint):
                q.SetActive(False)
                removed += 1
                continue
            for api in apis:
                if q.HasAPI(api):
                    q.RemoveAPI(api)
                    removed += 1
        return removed
