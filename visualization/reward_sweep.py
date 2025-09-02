# tests/reward_sweep.py
# Runs PPO on multiple examples with five reward designs (R1–R5) and
# saves learning-curve plots + CSVs per example.
#
# Usage:
#   python tests/reward_sweep.py --examples Example1 Example3 --episodes 800 --model none
#
# Notes:
# - Uses your existing GearEnv, PPOAgent, Processor, Pathfinder, Renderer, etc.
# - Wraps the environment reward *without* touching your core env code.
# - If some info[...] keys are missing, safe defaults are used.

import os, sys, json, math, argparse
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- Project imports (same trick as in test_gear_generation.py) ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.processor import Processor
from pathfinding.finder import Pathfinder
from geometry_env.env import GearEnv
from rl_agent.agents.ppo_agent import PPOAgent
from visualization.renderer import Renderer
from common.data_models import Point

# --------------------------
# Helpers
# --------------------------
def _json_np_default(o):
    import numpy as _np
    if isinstance(o, _np.integer): return int(o)
    if isinstance(o, _np.floating): return float(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    if isinstance(o, torch.Tensor):
        t = o.detach().cpu()
        return t.item() if t.ndim == 0 else t.tolist()
    if hasattr(o, "item"):
        try: return o.item()
        except Exception: pass
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def _to_point(p):
    if hasattr(p, "x") and hasattr(p, "y"):
        return Point(x=float(p.x), y=float(p.y))
    if isinstance(p, dict) and "x" in p and "y" in p:
        return Point(x=float(p["x"]), y=float(p["y"]))
    return Point(x=float(p[0]), y=float(p[1]))

# --------------------------
# Reward wrapper
# --------------------------
@dataclass
class RewardCfg:
    mode: str   # R1..R5
    target_perf: float = 75.0
    # weights (tune if you want)
    alpha: float = 1.0   # primary objective weight
    beta: float = 1.0
    gamma: float = 1.0
    mu: float = 0.5      # non-convex term weight
    kappa: float = 20.0  # failure penalty
    lamb_backlash: float = 2.0
    lamb_contact: float = 2.0
    lamb_overlap: float = 1.0
    eta_ratio: float = 15.0
    eta_path: float = 10.0
    eta_footprint: float = 0.1
    min_chain_len: int = 3

class RewardWrapper:
    """
    Wraps a GearEnv-like env to transform reward according to RewardCfg.
    Assumes env.step(action) -> (state, reward, terminated, truncated, info)
    We re-compute 'reward' from info safely using defaults.
    """
    def __init__(self, env, cfg: RewardCfg):
        self.env = env
        self.cfg = cfg
        self.episode_len = 0
        self.accum_final_bonus = 0.0
        self.last_place_ok = False

    def reset(self, *args, **kwargs):
        self.episode_len = 0
        self.accum_final_bonus = 0.0
        return self.env.reset(*args, **kwargs)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def step(self, action):
        state, _, terminated, truncated, info = self.env.step(action)
        self.episode_len += 1

        # Pull safe values from info (use defaults if not present)
        torque_gain = float(info.get("torque_gain", 0.0))
        volume = float(info.get("volume", 1.0))
        place_ok = bool(info.get("place_ok", info.get("success", False))) and not info.get("error")
        backlash_viol = float(info.get("backlash_violation", 0.0))
        contact_viol = float(info.get("contact_violation", 0.0))
        overlap_viol = float(info.get("overlap_violation", 0.0))
        center_dist = float(info.get("center_dist", 0.0))
        chain_len = int(info.get("chain_len", info.get("gears_count", 0)))
        ratio_hit = bool(info.get("ratio_hit", False))
        path_completed = bool(info.get("path_completed", False))
        footprint = float(info.get("footprint", volume))

        # Primary shaped objective
        primary = (torque_gain / max(volume, 1e-6))

        # hinge penalties
        hinge = (
            self.cfg.lamb_backlash * max(0.0, backlash_viol) +
            self.cfg.lamb_contact  * max(0.0, contact_viol)  +
            self.cfg.lamb_overlap  * max(0.0, overlap_viol)
        )

        r = 0.0
        m = self.cfg.mode.upper()

        if m == "R1":
            # Dense + small penalties
            r = self.cfg.alpha * primary - hinge

        elif m == "R2":
            # Sparse: success-only, otherwise heavy penalty
            if place_ok:
                r = self.cfg.beta * primary - hinge
            else:
                r = -self.cfg.kappa

        elif m == "R3":
            # Multi-objective + delayed credit
            r = self.cfg.gamma * primary - hinge
            # accumulate terminal bonuses to add only at the end
            self.accum_final_bonus += 0.0  # nothing stepwise; terminal adds bonuses below

        elif m == "R4":
            # Deceptive non-convex shaping
            osc = math.sin(0.07 * center_dist) - 0.5 * math.sin(0.23 * center_dist)
            r = self.cfg.alpha * primary - hinge - self.cfg.mu * osc

        elif m == "R5":
            # Hardest: delayed & non-convex & success-only and chain length gate
            if place_ok and chain_len >= self.cfg.min_chain_len:
                osc = math.sin(0.07 * center_dist) - 0.5 * math.sin(0.23 * center_dist)
                r = self.cfg.alpha * primary - hinge - self.cfg.mu * osc
            else:
                r = 0.0  # no credit unless placement ok and chain is long enough

        # terminal bonuses for R3/R5
        if terminated or truncated:
            bonus = 0.0
            if m in ("R3", "R5"):
                if ratio_hit:      bonus += self.cfg.eta_ratio
                if path_completed: bonus += self.cfg.eta_path
                bonus -= self.cfg.eta_footprint * footprint
            r += bonus

        return state, float(r), terminated, truncated, info

    def close(self):
        return self.env.close()

# --------------------------
# Minimal PPO training loop
# --------------------------
def train_one(env, episodes=400, max_steps=64, lr=3e-4, gamma=0.99, clip_eps=0.2, update_every=2048):
    """
    Lightweight PPO training to produce a reward curve.
    Uses your PPOAgent class.
    """
    state_dim = env.observation_space.shape[0]
    action_dims = env.action_space.nvec

    agent = PPOAgent(state_dim=state_dim, action_dims=action_dims, lr=lr, gamma=gamma, clip_epsilon=clip_eps)
    rewards_curve = []
    memory_states, memory_actions, memory_logprobs, memory_rewards, memory_dones = [], [], [], [], []
    timesteps = 0

    for ep in range(episodes):
        s, _ = env.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            a, logp = agent.act(s)
            s2, r, term, trunc, info = env.step(a)
            ep_reward += float(r)

            # Store transition (PPOAgent typically has a memory structure; we’ll keep lists)
            memory_states.append(s)
            memory_actions.append(a)
            memory_logprobs.append(logp)
            memory_rewards.append(r)
            memory_dones.append(term or trunc)

            s = s2
            timesteps += 1

            if term or trunc:
                break

        rewards_curve.append(ep_reward)

        # update PPO on schedule
        if timesteps >= update_every:
            agent.update(memory_states, memory_actions, memory_logprobs, memory_rewards, memory_dones)
            memory_states, memory_actions, memory_logprobs, memory_rewards, memory_dones = [], [], [], [], []
            timesteps = 0

    # one last update if any leftovers
    if memory_states:
        agent.update(memory_states, memory_actions, memory_logprobs, memory_rewards, memory_dones)

    env.close()
    return np.array(rewards_curve, dtype=float)

# --------------------------
# Build env from example
# --------------------------
def build_env_for_example(example_name, base_dir):
    data_dir = os.path.join(base_dir, "data")
    interm_dir = os.path.join(base_dir, "data", "intermediate")
    out_dir = os.path.join(base_dir, "output", example_name)
    os.makedirs(interm_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    input_img_path = os.path.join(data_dir, f"{example_name}.png")
    input_constraints_path = os.path.join(data_dir, f"{example_name}_constraints.json")
    processed_json_path = os.path.join(interm_dir, f"{example_name}_processed.json")

    # Preprocess
    try:
        Processor.process_input(input_img_path, input_constraints_path, processed_json_path)
    except Exception as e:
        print(f"[{example_name}] Preprocessing warning: {e}")

    # Pathfinding (cache)
    path_json_path = os.path.join(out_dir, "path.json")
    if os.path.exists(path_json_path):
        with open(path_json_path, "r") as f:
            optimal_path = json.load(f)
        print(f"[{example_name}] Loaded existing path with {len(optimal_path)} points.")
    else:
        finder = Pathfinder()
        optimal_path = finder.find_path(processed_json_path)
        if not optimal_path:
            raise RuntimeError(f"[{example_name}] Could not find path.")
        with open(path_json_path, "w") as f:
            json.dump(optimal_path, f, indent=2, default=_json_np_default)
        print(f"[{example_name}] Generated new path with {len(optimal_path)} points.")

    # Base env config (tune as needed)
    env_config = {
        "json_path": processed_json_path,
        "path": optimal_path,
        "module": 1.0,
        "clearance_margin": 0.5,
        "initial_gear_teeth": 20,
        "target_torque": 2.0,
        "torque_weight": 0.6,
        "space_weight": 0.3,
        "weight_penalty": 0.1,
    }
    env = GearEnv(env_config)
    return env, out_dir

# --------------------------
# Plot + CSV helpers
# --------------------------
def save_curves(example_name, out_dir, curves_dict):
    # CSV
    csv_path = os.path.join(out_dir, f"{example_name}_reward_sweep.csv")
    with open(csv_path, "w") as f:
        # header
        modes = list(curves_dict.keys())
        max_len = max(len(v) for v in curves_dict.values())
        f.write("episode," + ",".join(modes) + "\n")
        for i in range(max_len):
            row = [str(i+1)]
            for m in modes:
                arr = curves_dict[m]
                row.append("" if i >= len(arr) else f"{arr[i]:.6f}")
            f.write(",".join(row) + "\n")
    print(f"[{example_name}] Saved CSV → {csv_path}")

    # Plot
    plt.figure(figsize=(12, 6))
    for m, arr in curves_dict.items():
        plt.plot(np.arange(1, len(arr)+1), arr, label=m)
    plt.axhline(75, linestyle="--", label="Target performance")
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward per Episode")
    plt.title(f"Learning Slowdown under Increasing Reward Complexity — {example_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{example_name}_reward_sweep.png")
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"[{example_name}] Saved plot → {fig_path}")

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", nargs="+", required=True, help="Example names, e.g., Example1 Example3")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes per reward mode")
    parser.add_argument("--max_steps", type=int, default=64, help="Max steps per episode")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    torch.set_default_tensor_type("torch.FloatTensor")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU to keep runs comparable

    reward_modes = {
        "R1 Dense": RewardCfg(mode="R1"),
        "R2 Sparse": RewardCfg(mode="R2"),
        "R3 Multi+Delayed": RewardCfg(mode="R3"),
        "R4 Deceptive": RewardCfg(mode="R4"),
        "R5 Hardest": RewardCfg(mode="R5"),
    }

    for ex in args.examples:
        print(f"\n========== Running reward sweep for {ex} ==========")
        curves = {}
        for label, cfg in reward_modes.items():
            env, out_dir = build_env_for_example(ex, BASE_DIR)
            env_wrapped = RewardWrapper(env, cfg)
            rewards = train_one(env_wrapped, episodes=args.episodes, max_steps=args.max_steps)
            curves[label] = rewards
        save_curves(ex, out_dir, curves)

if __name__ == "__main__":
    main()
