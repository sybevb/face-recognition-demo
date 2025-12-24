"""
Plot Hailo CSV logs over time with EMA smoothing.

Usage:
  python3 plot_logs.py --csv logs/prob_reg_20251224_141336.csv

If --csv is omitted, the newest logs/prob_reg_*.csv is used.
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CLASS_NAMES = ["No view", "Pipe"]
REGRESSION_NAMES = ["Spray", "Under water", "Dirty lens"]


def smooth_series_ema(series, alpha=0.2):
    out = []
    ema = None
    for v in series:
        ema = v if ema is None else alpha * v + (1 - alpha) * ema
        out.append(ema)
    return out


def enforce_min_duration(t, flags, min_sec):
    if len(flags) == 0:
        return flags

    segments = []
    start = 0
    current = flags[0]
    for i in range(1, len(flags) + 1):
        if i == len(flags) or flags[i] != current:
            segments.append({"start": start, "end": i - 1, "value": current})
            if i < len(flags):
                start = i
                current = flags[i]

    def seg_duration(seg):
        return float(t.iloc[seg["end"]] - t.iloc[seg["start"]])

    i = 0
    while i < len(segments):
        if seg_duration(segments[i]) < min_sec:
            if i > 0:
                segments[i - 1]["end"] = segments[i]["end"]
                segments.pop(i)
                i = max(i - 1, 0)
                continue
            if i + 1 < len(segments):
                segments[i + 1]["start"] = segments[i]["start"]
                segments.pop(i)
                continue
        i += 1

    smoothed = flags[:]
    for seg in segments:
        for idx in range(seg["start"], seg["end"] + 1):
            smoothed[idx] = seg["value"]
    return smoothed


def find_latest_csv(log_dir: Path) -> Path:
    candidates = sorted(
        log_dir.glob("prob_reg_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        raise FileNotFoundError(f"No prob_reg_*.csv files found in {log_dir}")
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="", help="Path to CSV log")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing CSV logs")
    parser.add_argument("--alpha", type=float, default=0.2, help="EMA smoothing alpha")
    args = parser.parse_args()

    if args.csv:
        csv_path = Path(args.csv)
    else:
        csv_path = find_latest_csv(Path(args.log_dir))

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    t = df["timestamp_sec"]

    cls0 = smooth_series_ema(df["cls_0"], alpha=args.alpha)
    cls1 = smooth_series_ema(df["cls_1"], alpha=args.alpha)
    reg0 = smooth_series_ema(df["reg_0"], alpha=args.alpha)
    reg1 = smooth_series_ema(df["reg_1"], alpha=args.alpha)
    reg2 = smooth_series_ema(df["reg_2"], alpha=args.alpha)

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    plt.plot(t, cls0, label=f"{CLASS_NAMES[0]} (ema)")
    plt.plot(t, cls1, label=f"{CLASS_NAMES[1]} (ema)")
    plt.plot(t, reg0, label=f"{REGRESSION_NAMES[0]} (ema)")
    plt.plot(t, reg1, label=f"{REGRESSION_NAMES[1]} (ema)")
    plt.plot(t, reg2, label=f"{REGRESSION_NAMES[2]} (ema)")
    plt.title("Model outputs over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Value / Probability")

    is_pipe_ok = [
        (c1 > 0.5) and (r0 < 0.5) and (r1 < 0.5) and (r2 < 0.5)
        for c1, r0, r1, r2 in zip(
            df["cls_1"], df["reg_0"], df["reg_1"], df["reg_2"]
        )
    ]
    is_pipe_ok = enforce_min_duration(t, is_pipe_ok, min_sec=0.5)
    green_pct = (sum(is_pipe_ok) / len(is_pipe_ok) * 100.0) if len(is_pipe_ok) > 0 else 0.0
    plt.title(f"Model outputs over time (success: {green_pct:.1f}%)")

    # Background spans on main plot
    start_idx = 0
    current = is_pipe_ok[0] if len(is_pipe_ok) > 0 else False
    for i in range(1, len(is_pipe_ok) + 1):
        if i == len(is_pipe_ok) or is_pipe_ok[i] != current:
            color = "green" if current else "red"
            ax.axvspan(t.iloc[start_idx], t.iloc[i - 1], color=color, alpha=0.35)
            if i < len(is_pipe_ok):
                start_idx = i
                current = is_pipe_ok[i]

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
