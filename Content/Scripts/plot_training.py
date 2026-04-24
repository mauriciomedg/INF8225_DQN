import os
import sys
import pandas as pd
import matplotlib.pyplot as plt


LOG_PATH = "training_log.csv"
ROLLING_WINDOW = 50


def require_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Make sure your training log CSV exists."
        )


def safe_rolling(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def plot_curve(
    x,
    y,
    y_smooth=None,
    title="",
    xlabel="",
    ylabel="",
    raw_label="Raw",
    smooth_label="Moving Average",
):
    plt.figure(figsize=(9, 5))
    plt.plot(x, y, alpha=0.35, label=raw_label)

    if y_smooth is not None:
        plt.plot(x, y_smooth, linewidth=2.0, label=smooth_label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()


def main():
    log_path = LOG_PATH
    if len(sys.argv) > 1:
        log_path = sys.argv[1]

    require_file(log_path)

    df = pd.read_csv(log_path)

    if df.empty:
        raise ValueError("The CSV file is empty.")

    if "episode" not in df.columns:
        df["episode"] = range(len(df))

    # Convert columns safely if present
    numeric_cols = [
        "episode",
        "episode_reward",
        "episode_length",
        "final_distance",
        "success",
        "epsilon",
        "avg_loss",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["episode"]).copy()

    x = df["episode"]

    # Reward plot
    if "episode_reward" in df.columns:
        reward_ma = safe_rolling(df["episode_reward"], ROLLING_WINDOW)
        plot_curve(
            x=x,
            y=df["episode_reward"],
            y_smooth=reward_ma,
            title="Episode Reward",
            xlabel="Episode",
            ylabel="Reward",
            raw_label="Episode reward",
            smooth_label=f"Reward MA{ROLLING_WINDOW}",
        )

    # Loss plot
    if "avg_loss" in df.columns:
        loss_ma = safe_rolling(df["avg_loss"], ROLLING_WINDOW)
        plot_curve(
            x=x,
            y=df["avg_loss"],
            y_smooth=loss_ma,
            title="Average Loss per Episode",
            xlabel="Episode",
            ylabel="Loss",
            raw_label="Avg loss",
            smooth_label=f"Loss MA{ROLLING_WINDOW}",
        )

    # Final distance plot
    if "final_distance" in df.columns:
        dist_ma = safe_rolling(df["final_distance"], ROLLING_WINDOW)
        plot_curve(
            x=x,
            y=df["final_distance"],
            y_smooth=dist_ma,
            title="Final Distance to Player",
            xlabel="Episode",
            ylabel="Distance",
            raw_label="Final distance",
            smooth_label=f"Distance MA{ROLLING_WINDOW}",
        )

    # Success rate plot
    if "success" in df.columns:
        success_ma = safe_rolling(df["success"], ROLLING_WINDOW)
        plt.figure(figsize=(9, 5))
        plt.plot(x, success_ma, linewidth=2.0, label=f"Success Rate MA{ROLLING_WINDOW}")
        plt.title("Success Rate")
        plt.xlabel("Episode")
        plt.ylabel("Success rate")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

    # Epsilon plot
    if "epsilon" in df.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(x, df["epsilon"], linewidth=2.0, label="Epsilon")
        plt.title("Exploration Schedule")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

    # Episode length plot
    if "episode_length" in df.columns:
        len_ma = safe_rolling(df["episode_length"], ROLLING_WINDOW)
        plot_curve(
            x=x,
            y=df["episode_length"],
            y_smooth=len_ma,
            title="Episode Length",
            xlabel="Episode",
            ylabel="Steps",
            raw_label="Episode length",
            smooth_label=f"Length MA{ROLLING_WINDOW}",
        )

    # Print some summary stats
    print("=== Training Summary ===")
    print(f"Number of episodes: {len(df)}")

    if "episode_reward" in df.columns:
        print(f"Final reward MA{ROLLING_WINDOW}: {safe_rolling(df['episode_reward'], ROLLING_WINDOW).iloc[-1]:.4f}")

    if "final_distance" in df.columns:
        print(f"Final distance MA{ROLLING_WINDOW}: {safe_rolling(df['final_distance'], ROLLING_WINDOW).iloc[-1]:.4f}")

    if "success" in df.columns:
        print(f"Final success rate MA{ROLLING_WINDOW}: {safe_rolling(df['success'], ROLLING_WINDOW).iloc[-1]:.4f}")

    if "avg_loss" in df.columns:
        print(f"Final loss MA{ROLLING_WINDOW}: {safe_rolling(df['avg_loss'], ROLLING_WINDOW).iloc[-1]:.6f}")

    plt.show()


if __name__ == "__main__":
    main()