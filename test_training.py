#!/usr/bin/env python3
"""
Minimal end-to-end test for RL training pipelines.
Tests MAPPO on Formation (multi-agent) and PPO on Hover (single-agent).
Each test runs in isolation to avoid simulation context issues.
"""
import sys
import os
import subprocess

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_test(algo_name, task_name, cfg_file, algo_cfg_file, ravel_obs=False):
    """Run a single training test in a subprocess."""
    env = os.environ.copy()
    code = f"""
import sys, os
sys.path.insert(0, {SCRIPT_DIR!r})

from omni_drones import init_simulation_app
simulation_app = init_simulation_app({{"headless": True}})

import torch
from omegaconf import OmegaConf
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

OmegaConf.register_new_resolver("eval", eval)

from omni_drones.envs.isaac_env import IsaacEnv
from omni_drones.learning import ALGOS
from omni_drones.utils.torchrl import SyncDataCollector, EpisodeStats
from omni_drones.utils.torchrl.transforms import ravel_composite
from torchrl.data import Composite
from torchrl.envs.utils import set_exploration_type, ExplorationType

algo_name = {algo_name!r}
task_name = {task_name!r}
cfg_file = {cfg_file!r}
algo_cfg_file = {algo_cfg_file!r}
ravel_obs = {ravel_obs}

sys.stderr.write(f"\\n=== Testing {{algo_name}} on {{task_name}} ===\\n")
sys.stderr.flush()

task_cfg = OmegaConf.load(cfg_file)
base_env = OmegaConf.load("cfg/base/env_base.yaml")
sim_base = OmegaConf.load("cfg/base/sim_base.yaml")
algo_cfg = OmegaConf.load(algo_cfg_file)
for key, default in [
    ("priv_actor", False), ("priv_critic", False),
    ("checkpoint_path", None), ("phase", "default"),
]:
    if key not in algo_cfg:
        algo_cfg[key] = default
task_merged = OmegaConf.merge(sim_base, base_env, task_cfg)
cfg = OmegaConf.create({{
    "headless": True,
    "task": task_merged,
    "sim": "${{task.sim}}",
    "env": "${{task.env}}",
    "viewer": {{"resolution": [960, 720], "eye": [8, 0., 6.], "lookat": [0., 0., 1.]}},
    "algo": algo_cfg,
    "seed": 0,
}})
OmegaConf.resolve(cfg)

env_class = IsaacEnv.REGISTRY[task_name]
base_env = env_class(cfg, headless=True)

transforms = [InitTracker()]
if ravel_obs:
    transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    try:
        obs_central = base_env.observation_spec["agents", "observation_central"]
        if isinstance(obs_central, Composite):
            transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))
    except (KeyError, TypeError):
        pass
env = TransformedEnv(base_env, Compose(*transforms)).train()
env.set_seed(cfg.seed)

policy = ALGOS[algo_name](
    cfg.algo,
    env.observation_spec,
    env.action_spec,
    env.reward_spec,
    device=base_env.device
)
sys.stderr.write(f"  Policy: {{type(policy).__name__}}\\n")
sys.stderr.flush()

frames_per_batch = env.num_envs * int(cfg.algo.train_every)
total_frames = frames_per_batch * 2

stats_keys = [
    k for k in base_env.observation_spec.keys(True, True)
    if isinstance(k, tuple) and k[0]=="stats"
]
episode_stats = EpisodeStats(stats_keys)

collector = SyncDataCollector(
    env,
    policy=policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    device=cfg.sim.device,
    return_same_td=True,
)

env.train()
for i, data in enumerate(collector):
    sys.stderr.write(f"  Iteration {{i}}: collected data\\n")
    sys.stderr.flush()
    episode_stats.add(data.to_tensordict())
    info = policy.train_op(data.to_tensordict())
    sys.stderr.write(f"  Train op: {{list(info.keys())[:5]}}...\\n")
    sys.stderr.flush()

sys.stderr.write(f"  {{algo_name}} on {{task_name}}: PASSED\\n")
sys.stderr.flush()
simulation_app.close()
"""
    result = subprocess.run(
        [PYTHON, "-c", code],
        cwd=SCRIPT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    # Print stderr (progress/debug output)
    if result.stderr:
        sys.stderr.write(result.stderr)
        sys.stderr.flush()
    if result.returncode != 0:
        sys.stderr.write(f"\nFAILED: {algo_name} on {task_name} (exit code {result.returncode})\n")
        sys.stderr.flush()
        return False
    return True


# Test PPO on Hover (single-agent)
ok1 = run_test("ppo", "Hover", "cfg/task/Hover.yaml", "cfg/algo/mappo.yaml")

# Test MAPPO on Formation (multi-agent)
ok2 = run_test("mappo", "Formation", "cfg/task/Formation.yaml", "cfg/algo/mappo.yaml", ravel_obs=True)

if ok1 and ok2:
    sys.stderr.write("\nAll training pipeline tests PASSED\n")
    sys.stderr.flush()
    sys.exit(0)
else:
    sys.stderr.write("\nSome training pipeline tests FAILED\n")
    sys.stderr.flush()
    sys.exit(1)
