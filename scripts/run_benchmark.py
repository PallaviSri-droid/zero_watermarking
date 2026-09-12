from zero_watermarking.benchmark import run_benchmark
from zero_watermarking.visualization import plot_pareto, save_summary_bars


def main() -> None:
    frame = run_benchmark()
    print(frame.to_string(index=False))
    save_summary_bars(frame)
    plot_pareto(frame)


if __name__ == "__main__":
    main()
