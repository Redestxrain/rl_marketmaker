import numpy as np
from scipy import stats

def compute_all_metrics(rewards, label="Agent"):
    r        = np.array(rewards, dtype=float)
    std_pnl  = float(r.std())
    sharpe   = float(r.mean() / std_pnl) if std_pnl > 1e-9 else 0.0
    downside = r[r < 0]
    sortino  = (float(r.mean() / downside.std())
                if len(downside) > 0 and downside.std() > 1e-9
                else float("inf"))
    cumul  = np.cumsum(r)
    peak   = np.maximum.accumulate(cumul)
    max_dd = float((peak - cumul).max())
    return {
        "agent":        label,
        "n_episodes":   len(r),
        "mean_pnl":     round(float(r.mean()), 4),
        "std_pnl":      round(std_pnl, 4),
        "sharpe":       round(sharpe, 4),
        "sortino":      round(sortino, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate":     round(float((r > 0).mean()), 4),
        "median":       round(float(np.median(r)), 4),
        "p5":           round(float(np.percentile(r, 5)), 4),
        "p95":          round(float(np.percentile(r, 95)), 4),
    }

def compare_agents(results):
    print("\n" + "=" * 75)
    print(f"{"BENCHMARK COMPARISON":^75}")
    print("=" * 75)
    print(f"{"Agent":<26} {"Mean PnL":>10} {"Sharpe":>8} {"Sortino":>8} {"MaxDD":>8} {"WinRate":>8}")
    print("-" * 75)
    all_metrics = {}
    for label, rewards in results.items():
        m = compute_all_metrics(rewards, label)
        all_metrics[label] = m
        print(f"{label:<26} {m['mean_pnl']:>10.4f} {m['sharpe']:>8.4f} "
              f"{m['sortino']:>8.4f} {m['max_drawdown']:>8.4f} {m['win_rate']:>8.3f}")
    agents = list(results.keys())
    if len(agents) > 1:
        print("\n--- Wilcoxon signed-rank test vs first agent ---")
        base = np.array(results[agents[0]], dtype=float)
        for label in agents[1:]:
            other = np.array(results[label], dtype=float)
            n     = min(len(base), len(other))
            _, p  = stats.wilcoxon(base[:n], other[:n])
            sig   = "✓ significant" if p < 0.05 else "✗ not significant"
            print(f"  {label:<24} vs {agents[0]}: p={p:.4f}  {sig}")
    print("=" * 75)
    return all_metrics

def inventory_rmse(inventory_series):
    return float(np.sqrt(np.mean(np.array(inventory_series, dtype=float) ** 2)))
