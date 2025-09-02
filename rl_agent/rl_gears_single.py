# rl_gears_single.py
# One-file version: simulator + reward + env + demo figures
# - Heavy penalties for non-meshing/invalid states (hard constraints)
# - Exposes snap_error to visualize continuous path vs discrete teeth
# - Path is adjustable; boundary rejection enforced
# - Minimal "env" wrapper computing reward from simulator info
#
# No external RL libs required. Matplotlib only (Agg backend).

import math, random, os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Basic geometry & gears
# =========================

@dataclass
class Point:
    x: float
    y: float

@dataclass
class GearSet:
    id: str
    center: Point
    teeth_count: List[int]     # [driven, driving] or [z] when equal
    module: float = 1.0

    @property
    def driven_teeth(self) -> int:
        return int(self.teeth_count[0])

    @property
    def driving_teeth(self) -> int:
        return int(self.teeth_count[-1])

    @property
    def driven_radius(self) -> float:
        return 0.5 * self.module * self.driven_teeth

    @property
    def driving_radius(self) -> float:
        return 0.5 * self.module * self.driving_teeth

    @property
    def radii(self) -> List[float]:
        r = [self.driven_radius]
        if self.driving_teeth != self.driven_teeth:
            r.append(self.driving_radius)
        elif self.driving_radius not in r:
            r.append(self.driving_radius)
        # unique positive
        out, seen = [], set()
        for ri in r:
            if ri > 0 and abs(ri) not in seen:
                out.append(ri); seen.add(abs(ri))
        return out

class GearFactory:
    def __init__(self, module: float = 1.0):
        self.module = module

    def pitch_radius(self, z: int) -> float:
        return 0.5 * self.module * int(z)

    def create_gear(self, gear_id: str, center: Tuple[float, float], num_teeth: List[int]) -> GearSet:
        """
        Build a GearSet. GearSet dataclass expects field name `id`, not `gear_id`.
        """
        if len(num_teeth) == 1:
            num_teeth = [num_teeth[0], num_teeth[0]]
        return GearSet(
            id=gear_id,                          # <-- FIXED here
            center=Point(*center),
            teeth_count=[int(num_teeth[0]), int(num_teeth[-1])],
            module=self.module,
        )


# =========================
# Polyline / boundary utils
# =========================

def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def polyline_length(path: List[Point]) -> float:
    return sum(distance(path[i], path[i+1]) for i in range(len(path)-1))

def point_on_path_at_s(path: List[Point], s: float) -> Optional[Point]:
    if s < 0: return None
    total = polyline_length(path)
    if s >= total - 1e-12:
        return path[-1]
    rem = s
    for i in range(len(path)-1):
        p1, p2 = path[i], path[i+1]
        seg = distance(p1, p2)
        if rem <= seg:
            t = 0.0 if seg == 0 else rem/seg
            return Point(p1.x + t*(p2.x - p1.x), p1.y + t*(p2.y - p1.y))
        rem -= seg
    return None

def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay, bx, by, px, py = a.x, a.y, b.x, b.y, p.x, p.y
    abx, aby = bx-ax, by-ay
    ab2 = abx*abx + aby*aby
    if ab2 == 0: return math.hypot(px-ax, py-ay)
    t = max(0.0, min(1.0, ((px-ax)*abx + (py-ay)*aby)/ab2))
    qx, qy = ax + t*abx, ay + t*aby
    return math.hypot(px-qx, py-qy)

def distance_to_boundary(poly: List[Point], p: Point) -> float:
    return min(point_to_segment_distance(p, poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly)))

def winding_contains(poly: List[Point], p: Point) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        a = poly[i]; b = poly[(i+1)%n]
        if ((a.y > p.y) != (b.y > p.y)) and (p.x < (b.x-a.x)*(p.y-a.y)/((b.y-a.y) if (b.y-a.y)!=0 else 1e-12) + a.x):
            inside = not inside
    return inside


# =========================
# Simulator (your logic)
# =========================

class GearTrainSimulator:
    """
    Your simulator logic, enforcing:
      - Tangency at placement: prev.driving ↔ new.driven
      - Hard constraints: boundary + non-overlap (except the meshing pair)
      - Exposes snap_error for RL reward/visualization
    """
    def __init__(self,
                 path: List[List[float]],
                 input_shaft: Tuple[float,float],
                 output_shaft: Tuple[float,float],
                 boundaries: List[List[float]],
                 module: float = 1.0,
                 clearance_margin: float = 0.5,
                 backlash: float = 0.0):
        self.path = [Point(*p) for p in path]
        self.boundaries = [Point(*p) for p in boundaries]
        self.input_shaft = Point(*input_shaft)
        self.output_shaft = Point(*output_shaft)
        self.module = module
        self.clearance_margin = clearance_margin
        self.backlash = backlash
        self.factory = GearFactory(module=module)

        # Input/output gears sized by remaining space to boundary (simple demo)
        self.input_gear = self._max_gear_at(self.input_shaft, "gear_input")
        self.output_gear = self._max_gear_at(self.output_shaft, "gear_output")

        self.reset()

    # ---------- lifecycle ----------
    def reset(self):
        self.gears: List[GearSet] = [self.input_gear, self.output_gear]
        self.last_gear: GearSet = self.input_gear
        self.s: float = self._project_to_path_s(self.input_shaft)
        self._prev_s = self.s

    # ---------- helpers ----------
    def _max_gear_at(self, center: Point, gid: str) -> GearSet:
        r = max(0.0, distance_to_boundary(self.boundaries, center) - self.clearance_margin)
        # choose teeth that fit that radius (module * z / 2 <= r) → z <= 2r/module
        z = max(8, int(2.0 * r / self.module))
        return self.factory.create_gear(gid, (center.x, center.y), [z, z])

    def _project_to_path_s(self, p: Point) -> float:
        # project p to closest point on path; return arclength s to that projection
        best_s = 0.0
        best_d = float("inf")
        acc = 0.0
        for i in range(len(self.path)-1):
            a, b = self.path[i], self.path[i+1]
            abx, aby = b.x-a.x, b.y-a.y
            ab2 = abx*abx + aby*aby
            if ab2 == 0:
                d = distance(p, a)
                if d < best_d:
                    best_d, best_s = d, acc
                continue
            t = max(0.0, min(1.0, ((p.x-a.x)*abx + (p.y-a.y)*aby)/ab2))
            qx, qy = a.x + t*abx, a.y + t*aby
            d = math.hypot(p.x-qx, p.y-qy)
            if d < best_d:
                best_d = d
                best_s = acc + math.hypot(qx-a.x, qy-a.y)
            acc += math.hypot(b.x-a.x, b.y-a.y)
        return best_s

    def _path_total_length(self) -> float:
        return polyline_length(self.path)

    def _snap_along_path_to_distance(self, s_guess: float, target: Point, required_dist: float,
                                     half_window: float = 25.0, tol: float = 0.15, iters: int = 30):
        # Solve f(s) = distance(path(s), target) - required_dist = 0 by bisection if bracketed;
        # else accept closest end/guess if within tol.
        L = self._path_total_length()
        a = max(0.0, s_guess - half_window)
        b = min(L,    s_guess + half_window)
        def f(s):
            q = point_on_path_at_s(self.path, s)
            if q is None: return 1e9
            return distance(q, target) - required_dist
        fa, fb = f(a), f(b)
        if fa*fb > 0:
            cand = [(abs(fa), a), (abs(fb), b), (abs(f(s_guess)), s_guess)]
            cand.sort()
            if cand[0][0] <= tol:
                s_star = cand[0][1]; q = point_on_path_at_s(self.path, s_star)
                return s_star, q, f(s_star)
            return None, None, None
        for _ in range(iters):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) <= tol:
                q = point_on_path_at_s(self.path, m)
                return m, q, fm
            if fa*fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        m = 0.5*(a+b); fm = f(m)
        if abs(fm) <= tol:
            q = point_on_path_at_s(self.path, m)
            return m, q, fm
        return None, None, None

    def _max_pitch_radius(self, g: GearSet) -> float:
        return max(g.radii) if g.radii else 0.0

    def _valid_center_with_margin_for_new(self, new_gear: GearSet, tol: float = 0.2) -> Tuple[bool, str]:
        """
        Validate the new gear center against:
          - Previous gear must be tangent: prev.driving + new.driven (within tol/backlash)
          - Other existing gears must NOT overlap (max pitch radii + clearance)
          - Must be inside boundary with wall clearance
        """
        prev = self.last_gear
        d_prev = distance(prev.center, new_gear.center)
        target_prev = float(prev.driving_radius) + float(new_gear.driven_radius) - float(self.backlash)
        err = d_prev - target_prev
        if err < -tol:  # interference
            return False, f"prev overlap (err={err:.3f})"
        if err >  tol:  # not tangent / too far
            return False, f"prev too far (err={err:.3f})"

        # against others
        new_max = self._max_pitch_radius(new_gear)
        for g in self.gears:
            if g is prev:
                continue
            need = self._max_pitch_radius(g) + new_max + self.clearance_margin
            if distance(g.center, new_gear.center) < need - 1e-9:
                return False, f"overlap with {g.id}"

        # boundary checks
        if not winding_contains(self.boundaries, new_gear.center):
            return False, "outside boundary"
        if distance_to_boundary(self.boundaries, new_gear.center) < (new_max + self.clearance_margin) - 1e-9:
            return False, "too close to boundary"
        return True, ""

    def _push_snap_error_to_info(self, info: Dict[str,Any], err_value: Optional[float]) -> Dict[str,Any]:
        info = dict(info) if info is not None else {}
        try:
            val = float(abs(err_value)) if err_value is not None else 0.0
        except Exception:
            val = 0.0
        info.setdefault("snap_error", val)
        return info

    # ---------- main step ----------
    def step(self, action: Tuple[int,int]) -> Tuple[Dict[str,float], float, bool, Dict[str,Any]]:
        """
        action = (driven_teeth, driving_teeth)
        1) advance s by required meshing distance (prev.driving + new.driven - backlash)
        2) snap s along path to enforce tangency with previous
        3) validate hard constraints (no soft penalties here)
        4) if near output, (optionally) snap to output
        """
        driven_teeth, driving_teeth = int(action[0]), int(action[1])
        new_teeth = [driven_teeth, driving_teeth]

        # advance guess
        meshing_required = self.last_gear.driving_radius + self.factory.pitch_radius(driven_teeth) - self.backlash
        s_guess = self.s + meshing_required

        # snap to satisfy prev.driving ↔ new.driven
        s_star, p_star, err_prev = self._snap_along_path_to_distance(
            s_guess=s_guess,
            target=self.last_gear.center,
            required_dist=meshing_required,
            half_window=25.0, tol=0.2
        )
        if s_star is None or p_star is None:
            # cannot find tangency along path near guess → reject (hard)
            info = {"error": "no tangency along path",
                    "is_valid": False,
                    "mesh_all_tangent": False,
                    "non_tangent_pairs": 1}
            info = self._push_snap_error_to_info(info, None)
            return self._get_state(), 0.0, False, info

        # provisional gear at snapped center
        new_gear = self.factory.create_gear(
            gear_id=f"gear_{len(self.gears)-1}",
            center=(p_star.x, p_star.y),
            num_teeth=new_teeth
        )

        ok, why = self._valid_center_with_margin_for_new(new_gear, tol=0.2)
        if not ok:
            # hard rejection with reason; RL will get heavy negative from reward
            info = {"error": why,
                    "is_valid": False,
                    "mesh_all_tangent": False if "prev" in why else True,
                    "non_tangent_pairs": 0 if "prev" not in why else 1}
            info = self._push_snap_error_to_info(info, err_prev)
            return self._get_state(), 0.0, False, info

        # accept placement
        self.s = s_star
        self.gears.insert(-1, new_gear)
        self.last_gear = new_gear

        # info for reward
        info = {"mesh_all_tangent": True,
                "non_tangent_pairs": 0,
                "outside_boundary": False,
                "boundary_clearance_ok": True,
                "overlap_count": 0,
                "is_valid": True}
        info = self._push_snap_error_to_info(info, err_prev)

        # check output tangency
        req_out = self.last_gear.driving_radius + self.output_gear.driven_radius - self.backlash
        d_out = distance(self.last_gear.center, self.output_gear.center)
        if abs(d_out - req_out) <= 0.25:
            info["success"] = "meshed to output"
            return self._get_state(), 0.0, True, info

        # if near end, try to snap to output too
        if (self._path_total_length() - self.s) < 30.0:
            s2, p2, err_out = self._snap_along_path_to_distance(
                s_guess=self.s, target=self.output_gear.center, required_dist=req_out,
                half_window=25.0, tol=0.3
            )
            if s2 is not None and p2 is not None:
                self.s = s2
                self.gears[-2] = self.factory.create_gear(
                    gear_id=self.last_gear.id,
                    center=(p2.x, p2.y),
                    num_teeth=self.last_gear.teeth_count
                )
                self.last_gear = self.gears[-2]
                info["success"] = f"snapped to output (err={err_out:.3f})"
                info = self._push_snap_error_to_info(info, err_out)
                return self._get_state(), 0.0, True, info

        return self._get_state(), 0.0, False, info

    def _get_state(self) -> Dict[str,float]:
        return {
            "last_gear_center_x": self.last_gear.center.x,
            "last_gear_center_y": self.last_gear.center.y,
            "last_gear_driving_radius": self.last_gear.driving_radius,
            "target_gear_center_x": self.output_gear.center.x,
            "target_gear_center_y": self.output_gear.center.y,
            "target_gear_driven_radius": self.output_gear.driven_radius
        }


# =========================
# Reward (heavy penalties)
# =========================

def compute_reward(
    report: Dict[str,Any],
    target_torque: float = 2.0,
    torque_weight: float = 0.6,
    space_weight: float = 0.3,
    weight_penalty_coef: float = 0.1,
) -> float:
    """
    RL implications:
      - Heavily penalize non-meshing / invalid (outside/overlap/boundary) → HARD constraints
      - Small penalty for `snap_error` to *visualize* continuous(path) vs discrete(teeth) conflict
      - Keep shaping only for valid states (torque/space/mass placeholders)
    """
    # HARD constraints first
    outside_boundary      = bool(report.get("outside_boundary", False))
    boundary_clearance_ok = bool(report.get("boundary_clearance_ok", True))
    overlap_count         = int(report.get("overlap_count", 0))
    mesh_all_tangent      = bool(report.get("mesh_all_tangent", True))
    non_tangent_pairs     = int(report.get("non_tangent_pairs", 0))
    is_valid              = bool(report.get("is_valid", True))

    if outside_boundary:         return -200.0
    if not boundary_clearance_ok:return -150.0
    if overlap_count > 0:        return -150.0
    if not mesh_all_tangent or non_tangent_pairs > 0:
        k = max(1, non_tangent_pairs)
        return -100.0 - 25.0*(k-1)
    if not is_valid:             return -100.0

    # Valid state: shaping (placeholders—wire from your validator if you have them)
    torque_ratio = float(report.get("torque_ratio", target_torque))
    space_usage  = float(report.get("space_usage", 0.7))   # pretend decent space usage
    total_mass   = float(report.get("total_mass", 100.0))  # pretend mass

    torque_diff   = abs(torque_ratio - target_torque)
    torque_reward = math.exp(-torque_diff)
    space_reward  = max(0.0, min(1.0, space_usage))
    weight_pen    = total_mass * 0.01

    reward = torque_weight * torque_reward + space_weight * space_reward - weight_penalty_coef * weight_pen

    # Visualize continuous vs discrete: small penalty for snap_error
    snap_error = float(report.get("snap_error", 0.0))
    if snap_error > 0.0:
        reward -= 0.5 * snap_error

    return float(reward)


# =========================
# Env wrapper (minimal)
# =========================

class GearEnv:
    """
    Minimal wrapper around simulator:
      - converts action->teeth
      - calls simulator.step and compute_reward
      - returns observation, reward, done, info
    """
    def __init__(self, config: Dict[str,Any]):
        self.config = dict(config)
        self.simulator = GearTrainSimulator(
            path=config["path"],
            input_shaft=tuple(config["input_shaft"]),
            output_shaft=tuple(config["output_shaft"]),
            boundaries=config["boundary"],
            module=config.get("module", 1.0),
            clearance_margin=config.get("clearance_margin", 0.5),
            backlash=config.get("backlash", 0.0)
        )
        self.min_teeth = config.get("min_teeth", 8)

        self.target_torque   = config.get("target_torque", 2.0)
        self.torque_weight   = config.get("torque_weight", 0.6)
        self.space_weight    = config.get("space_weight", 0.3)
        self.weight_pen_coef = config.get("weight_penalty", 0.1)

    def reset(self):
        self.simulator.reset()
        return self._state_to_observation(self.simulator._get_state()), {}

    def _state_to_observation(self, state: Dict[str,float]) -> np.ndarray:
        return np.array([
            state["last_gear_center_x"], state["last_gear_center_y"], state["last_gear_driving_radius"],
            state["target_gear_center_x"], state["target_gear_center_y"], state["target_gear_driven_radius"]
        ], dtype=np.float32)

    def step(self, action: Tuple[int,int]):
        # convert from offsets to absolute teeth if needed
        try:
            driven_teeth  = int(action[0]) + self.min_teeth
            driving_teeth = int(action[1]) + self.min_teeth
        except Exception:
            driven_teeth, driving_teeth = int(action[0]), int(action[1])

        state, _sim_r, done, info = self.simulator.step((driven_teeth, driving_teeth))
        report = dict(info) if isinstance(info, dict) else {}
        reward = compute_reward(
            report=report,
            target_torque=self.target_torque,
            torque_weight=self.torque_weight,
            space_weight=self.space_weight,
            weight_penalty_coef=self.weight_pen_coef
        )
        obs = self._state_to_observation(state)
        return obs, float(reward), bool(done), False, info


# =========================
# Demo/figures
# =========================

def make_boundary_and_path(example: int = 1):
    # Adjustable path & boundary so you can show the continuous-vs-discrete effect
    if example == 1:
        boundary = [Point(-50,-22), Point(50,-22), Point(50,-2), Point(30,-2), Point(10,-8),
                    Point(-10,-8), Point(-30,-2), Point(-50,-2)]
        path = [Point(35,-1), Point(0,-4), Point(-15,-5), Point(-30,-6)]
        inp, outp = Point(35,-1), Point(-30,-6)
    else:
        boundary = [Point(-50,-22), Point(50,-22), Point(50, 3), Point(15, 3), Point(10,-6),
                    Point(-10,-6), Point(-15, 3), Point(-50, 3)]
        path = [Point(35,0), Point(20,-3), Point(0,-7), Point(-20,-6), Point(-30,-5)]
        inp, outp = Point(35,0), Point(-30,-5)
    return boundary, path, inp, outp

def draw_layout(ax, sim: GearTrainSimulator, title: str):
    bx = [p.x for p in sim.boundaries] + [sim.boundaries[0].x]
    by = [p.y for p in sim.boundaries] + [sim.boundaries[0].y]
    ax.plot(bx, by, linewidth=2)
    px = [p.x for p in sim.path]; py = [p.y for p in sim.path]
    ax.plot(px, py, linewidth=1, alpha=0.7)
    for g in sim.gears:
        cx, cy = g.center.x, g.center.y
        for r in g.radii:
            ax.add_patch(plt.Circle((cx, cy), r, fill=False))
        ax.plot([cx], [cy], "k.", ms=3)
        ax.text(cx, cy-1.2, g.id, fontsize=8, color="green")
    ax.set_aspect("equal", "box")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

def run_episode(env: GearEnv, policy, max_steps=8):
    obs, _ = env.reset()
    infos = []
    rewards = []
    for t in range(max_steps):
        a = policy(env)
        obs, r, done, trunc, info = env.step(a)
        infos.append(info)
        rewards.append(r)
        if done: break
    return infos, rewards

def random_policy(env: GearEnv):
    return (random.randint(0, 24), random.randint(0, 24))  # offsets from min_teeth

def masked_policy(env: GearEnv):
    # crude feasibility: ensure driven radius can reach from prev at next s
    sim = env.simulator
    guess = sim.s + sim.last_gear.driving_radius + sim.factory.pitch_radius(8)
    p = point_on_path_at_s(sim.path, guess)
    if p is None:
        return (4, 4)
    for zoff in range(0, 25):
        z = env.min_teeth + zoff
        need = sim.last_gear.driving_radius + sim.factory.pitch_radius(z) - sim.backlash
        if distance(p, sim.last_gear.center) + 1e-6 >= need:
            return (zoff, random.randint(0, 24))
    return (4, 4)

def make_figures():
    outdir = os.path.abspath("./rl_gears_output_single")
    os.makedirs(outdir, exist_ok=True)

    boundary, path, inp, outp = make_boundary_and_path(example=1)
    cfg = {
        "boundary": [[p.x, p.y] for p in boundary],
        "path": [[p.x, p.y] for p in path],
        "input_shaft": (inp.x, inp.y),
        "output_shaft": (outp.x, outp.y),
        "module": 1.0,
        "clearance_margin": 0.5,
        "backlash": 0.0,
        "min_teeth": 8,
        "target_torque": 2.0,
        "torque_weight": 0.6,
        "space_weight": 0.3,
        "weight_penalty": 0.1,
    }
    env = GearEnv(cfg)

    # Compare policies: random (more rejects) vs masked (fewer rejects)
    infos_soft, rewards_soft = run_episode(env, random_policy, max_steps=8)
    # reset whole env sim for a second episode visual clarity
    env = GearEnv(cfg)
    infos_hard, rewards_hard = run_episode(env, masked_policy, max_steps=8)

    # Layout after masked policy (more likely to place tangent gears)
    fig = plt.figure(figsize=(9,4), dpi=140)
    ax = fig.add_subplot(111)
    draw_layout(ax, env.simulator, "Hard constraints: always tangent to previous, boundary respected")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "layout_hard.png")); plt.close(fig)

    # Rewards timelines
    def plot_rewards(rr, name):
        fig = plt.figure(figsize=(6,3), dpi=140)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(rr)), rr, marker="o")
        ax.set_xlabel("step"); ax.set_ylabel("reward")
        ax.set_title(f"Per-step reward: {name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"rewards_{name}.png")); plt.close(fig)
    plot_rewards(rewards_soft, "random")
    plot_rewards(rewards_hard, "masked")

    # snap_error over steps (continuous vs discrete visualization)
    def plot_snap_errors(infos, name):
        se = [float(i.get("snap_error", 0.0)) for i in infos]
        fig = plt.figure(figsize=(6,3), dpi=140)
        ax = fig.add_subplot(111)
        ax.plot(np.arange(len(se)), se, marker="x")
        ax.set_xlabel("step"); ax.set_ylabel("|snap_error|")
        ax.set_title(f"Continuous path vs discrete teeth mismatch: {name}")
        ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"snap_error_{name}.png")); plt.close(fig)
    plot_snap_errors(infos_soft, "random")
    plot_snap_errors(infos_hard, "masked")

    print("Figures saved to:", outdir)

# =========================
# Main
# =========================

if __name__ == "__main__":
    make_figures()
