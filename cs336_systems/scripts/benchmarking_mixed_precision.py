from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

# Make the repository root importable when this script is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_systems.benchmark import BenchmarkConfig, run_configured_benchmark, summarize


MODEL_SWEEP = [
    ("small", "small", 768, 3072, 12, 12),
    ("medium", "medium", 1024, 4096, 24, 16),
    ("large", "large", 1280, 5120, 36, 20),
    ("xl", "xl", 1600, 6400, 48, 25),
    ("2.7B", "2.7b", 2560, 10240, 32, 32),
]

PRECISION_SWEEP = [
    ("fp32", False),
    ("bf16_mixed", True),
]

CSV_FIELDNAMES = [
    "model_size",
    "precision",
    "pass_type",
    "warmup_steps",
    "measured_steps",
    "mean_s",
    "std_s",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    """Parse the mixed-precision sweep configuration."""

    parser = argparse.ArgumentParser(
        description="Run assignment 1.1.5(c) BF16 mixed-precision benchmarking on a single CUDA GPU.",
    )
    parser.add_argument("--context-length", type=int, default=128, help="Sequence length used for every run.")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps for every benchmark run.")
    parser.add_argument("--measure-steps", type=int, default=10, help="Measured steps for every benchmark run.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for every run.")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size used for every run.")
    parser.add_argument("--rope-theta", type=float, default=10_000.0, help="RoPE theta used for every run.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Placeholder LR for config completeness.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed reused across forward and forward-backward runs.")
    parser.add_argument("--device", default="cuda", help="CUDA device to benchmark on, e.g. cuda or cuda:0.")
    return parser.parse_args()


def ensure_cuda(device_name: str) -> None:
    """Fail fast with a clear message when CUDA is unavailable."""

    device = torch.device(device_name)
    if device.type != "cuda":
        raise SystemExit("This script requires a CUDA device. Please pass --device cuda or cuda:<index>.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. This benchmark script must be run on a CUDA-capable machine.")


def is_oom_error(exc: BaseException) -> bool:
    """Return True when an exception corresponds to a CUDA OOM failure."""

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def cleanup_after_failure() -> None:
    """Clear CUDA allocator state so the sweep can continue after a failed run."""

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_config(
    *,
    model_size_key: str,
    mode: str,
    d_model: int,
    d_ff: int,
    num_layers: int,
    num_heads: int,
    args: argparse.Namespace,
    enable_bf16_autocast: bool,
) -> BenchmarkConfig:
    """Construct one benchmark config for the shared benchmark harness."""

    return BenchmarkConfig(
        model_size=model_size_key,
        mode=mode,
        device=args.device,
        seed=args.seed,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        rope_theta=args.rope_theta,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        enable_nvtx=False,
        enable_bf16_autocast=enable_bf16_autocast,
    )


def run_mode(config: BenchmarkConfig) -> tuple[str, str, dict[str, float] | None, list[float] | None]:
    """Run one benchmark mode and convert failures into structured status rows."""

    try:
        summary, step_times = run_configured_benchmark(config)
        return "ok", "", summary, step_times
    except Exception as exc:  # noqa: BLE001 - keep the sweep alive for all configs.
        cleanup_after_failure()
        status = "oom" if is_oom_error(exc) else "error"
        return status, str(exc), None, None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_result_row(
    *,
    model_size_label: str,
    precision_label: str,
    pass_type: str,
    warmup_steps: int,
    measure_steps: int,
    status: str,
    error_message: str,
    summary: dict[str, float] | None,
) -> dict[str, str | float | int]:
    """Build one CSV row for a successful or failed experiment."""

    return {
        "model_size": model_size_label,
        "precision": precision_label,
        "pass_type": pass_type,
        "warmup_steps": warmup_steps,
        "measured_steps": measure_steps,
        "mean_s": "" if summary is None else summary["mean_seconds"],
        "std_s": "" if summary is None else summary["stdev_seconds"],
        "status": status,
        "error_message": error_message,
    }


def format_experiment_label(model_size_label: str, precision_label: str, pass_type: str) -> str:
    """Format one experiment label for progress logging."""

    return f"model={model_size_label} precision={precision_label} pass={pass_type}"


def log_experiment_start(model_size_label: str, precision_label: str, pass_type: str) -> None:
    """Print which experiment is starting so long sweeps show live progress."""

    print(f"[run] {format_experiment_label(model_size_label, precision_label, pass_type)}", flush=True)


def log_experiment_result(
    model_size_label: str,
    precision_label: str,
    pass_type: str,
    status: str,
    summary: dict[str, float] | None,
    error_message: str,
    *,
    derived: bool = False,
) -> None:
    """Print the outcome of one experiment or derived backward estimate."""

    prefix = "[derived]" if derived else "[done]"
    label = format_experiment_label(model_size_label, precision_label, pass_type)
    if summary is not None:
        print(
            f"{prefix} {label} status={status} mean_s={summary['mean_seconds']:.6f} std_s={summary['stdev_seconds']:.6f}",
            flush=True,
        )
        return

    if error_message:
        print(f"{prefix} {label} status={status} error={error_message}", flush=True)
    else:
        print(f"{prefix} {label} status={status}", flush=True)


def run_sweep(args: argparse.Namespace) -> list[dict[str, str | float | int]]:
    """Run the full 1.1.5(c) sweep and return CSV-ready rows."""

    rows: list[dict[str, str | float | int]] = []

    for model_size_label, model_size_key, d_model, d_ff, num_layers, num_heads in MODEL_SWEEP:
        for precision_label, enable_bf16 in PRECISION_SWEEP:
            forward_config = make_config(
                model_size_key=model_size_key,
                mode="forward",
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_heads=num_heads,
                args=args,
                enable_bf16_autocast=enable_bf16,
            )
            fb_config = make_config(
                model_size_key=model_size_key,
                mode="forward-backward",
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_heads=num_heads,
                args=args,
                enable_bf16_autocast=enable_bf16,
            )

            log_experiment_start(model_size_label, precision_label, "forward")
            forward_status, forward_error, forward_summary, forward_step_times = run_mode(forward_config)
            log_experiment_result(model_size_label, precision_label, "forward", forward_status, forward_summary, forward_error)
            rows.append(
                make_result_row(
                    model_size_label=model_size_label,
                    precision_label=precision_label,
                    pass_type="forward",
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    status=forward_status,
                    error_message=forward_error,
                    summary=forward_summary,
                )
            )

            log_experiment_start(model_size_label, precision_label, "forward-backward")
            fb_status, fb_error, fb_summary, fb_step_times = run_mode(fb_config)
            log_experiment_result(model_size_label, precision_label, "forward-backward", fb_status, fb_summary, fb_error)
            backward_summary: dict[str, float] | None = None
            backward_status = "ok"
            backward_error = ""

            if forward_status != "ok":
                backward_status = forward_status
                backward_error = f"Backward estimate unavailable because forward benchmark failed: {forward_error}"
            elif fb_status != "ok":
                backward_status = fb_status
                backward_error = f"Backward estimate unavailable because forward-backward benchmark failed: {fb_error}"
            else:
                assert forward_step_times is not None
                assert fb_step_times is not None
                backward_step_times = [
                    forward_backward_s - forward_s
                    for forward_s, forward_backward_s in zip(forward_step_times, fb_step_times, strict=True)
                ]
                backward_summary = summarize(backward_step_times)

            log_experiment_result(
                model_size_label,
                precision_label,
                "backward",
                backward_status,
                backward_summary,
                backward_error,
                derived=True,
            )
            rows.append(
                make_result_row(
                    model_size_label=model_size_label,
                    precision_label=precision_label,
                    pass_type="backward",
                    warmup_steps=args.warmup_steps,
                    measure_steps=args.measure_steps,
                    status=backward_status,
                    error_message=backward_error,
                    summary=backward_summary,
                )
            )

    return rows


def write_results_csv(rows: list[dict[str, str | float | int]]) -> Path:
    """Write the sweep output CSV required by assignment 1.1.5(c)."""

    output_dir = PROJECT_ROOT / "cs336_systems" / "outputs" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmarking_mixed_precision.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def main() -> int:
    """Run the mixed-precision sweep and write its CSV output."""

    args = parse_args()
    ensure_cuda(args.device)
    rows = run_sweep(args)
    output_path = write_results_csv(rows)

    print(f"Wrote mixed-precision benchmark results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
