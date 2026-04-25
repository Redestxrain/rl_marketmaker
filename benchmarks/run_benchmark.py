import sys, os
sys.path.insert(0, "/content/rl_marketmaker")

import numpy as np
import json
from tqdm import tqdm

from simulation.market_gym import Market
from benchmarks.metrics    import compare_agents, compute_all_metrics

BASE_CONFIG = {
    "market_env":      "noise",
    "execution_agent": "sl_agent",
    "volume":          20,
    "terminal_time":   150,
    "time_delta":      15,
    "drop_feature":    "drift",
    "seed":            200,
}

def rollout_agent(agent_type, config, n_episodes):
    cfg     = {**config, "execution_agent": agent_type}
    rewards = []
    env     = Market(cfg)
    for _ in tqdm(range(n_episodes), desc=f"  {agent_type}", leave=True):
        _, info = env.reset()
        rewards.append(float(info["reward"]))
    return rewards

def rollout_rl_agent(model_path, config, n_episodes):
    import torch
    import gymnasium as gym
    from rl_files.actor_critic import BilateralAgentLogisticNormal

    rl_config  = {**config, "execution_agent": "rl_agent"}
    dummy_envs = gym.vector.SyncVectorEnv([lambda: Market(rl_config)])
    agent      = BilateralAgentLogisticNormal(dummy_envs)
    agent.load_state_dict(torch.load(model_path, map_location="cpu"))
    agent.eval()

    rl_env   = Market(rl_config)
    rewards  = []
    obs, _   = rl_env.reset()
    ep_count = 0

    with tqdm(total=n_episodes, desc="  PPO_Bilateral") as pbar:
        while ep_count < n_episodes:
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                bid_a, ask_a = agent.deterministic_action(obs_t)
                env_action   = (bid_a.squeeze(0).numpy(),
                                ask_a.squeeze(0).numpy())
            obs, _, terminated, _, info = rl_env.step(env_action)
            if terminated:
                rewards.append(float(info["reward"]))
                ep_count += 1
                pbar.update(1)
                obs, _ = rl_env.reset()
    return rewards

def run_full_benchmark(market_env="noise", n_episodes=200, model_path=None):
    config = {**BASE_CONFIG, "market_env": market_env}
    print(f"\n{chr(9472)*60}")
    print(f"  BENCHMARKING  |  env: {market_env.upper()}  |  episodes: {n_episodes}")
    print(f"{chr(9472)*60}")
    results = {}

    print("\n[1/4] BilateralTop (best bid+ask both sides)...")
    results["BilateralTop"]      = rollout_agent("top_agent",       config, n_episodes)

    print("\n[2/4] SubmitAndLeave (sell-only baseline)...")
    results["SubmitAndLeave"]    = rollout_agent("sl_agent",        config, n_episodes)

    print("\n[3/4] LinearSubmitLeave (sell-only baseline)...")
    results["LinearSubmitLeave"] = rollout_agent("linear_sl_agent", config, n_episodes)

    if model_path and os.path.exists(model_path):
        print("\n[4/4] PPO Bilateral RL agent...")
        results["PPO_Bilateral"] = rollout_rl_agent(model_path, config, n_episodes)
    else:
        print("\n[4/4] Skipping RL agent (no model path provided)")

    all_metrics = compare_agents(results)
    os.makedirs("benchmarks/results", exist_ok=True)
    save_path = f"benchmarks/results/{market_env}_benchmark.json"
    with open(save_path, "w") as f:
        json.dump({"config": config, "metrics": all_metrics,
                   "rewards": {k: v for k, v in results.items()}}, f, indent=2)
    print(f"\n✓ Saved → {save_path}")
    return all_metrics
