from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cs336_systems.benchmark import (  # noqa: E402
    MODEL_SIZE_PRESETS,
    BenchmarkConfig,
    autocast_context,
    make_batch,
    make_model,
    resolve_model_size_name,
    synchronize,
    validate_runtime,
)


DEFAULT_CONTEXT_LENGTHS = [128, 256, 512]
CSV_FIELDNAMES = [
    "model_size",
    "context_length",
    "batch_size",
    "mode",
    "mixed_precision",
    "warmup_steps",
    "measure_steps",
    "max_memory_allocated_bytes",
    "max_memory_allocated_mib",
    "max_memory_reserved_bytes",
    "max_memory_reserved_mib",
    "snapshot_path",
    "status",
    "error_message",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the memory profiling experiment script."""

    parser = argparse.ArgumentParser(
        description=(
            "Profile Transformer memory usage with PyTorch memory snapshots. "
            "By default this script runs the medium model across context lengths "
            "128, 256, and 512 for forward/train and FP32/BF16 configurations."
        ),
    )
    parser.add_argument(
        "--model-size",
        choices=("small", "medium", "large", "xl", "2.7b", "2.7B"),
        default="medium",
        help="Model size preset to profile. Defaults to medium for one-click runs.",
    )
    parser.add_argument("--context-length", type=int, default=128, help="Sequence length to profile when not sweeping all context lengths.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for random token inputs.")
    parser.add_argument(
        "--mode",
        choices=("forward", "train"),
        default=None,
        help="Run only one mode. If omitted, run both forward and full train-step profiling.",
    )
    parser.add_argument("--warmup-steps", type=int, default=3, help="Number of warmup steps before memory recording starts.")
    parser.add_argument("--measure-steps", type=int, default=10, help="Number of profiled steps after warmup.")
    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "If set, run only BF16 mixed precision. If disabled with --no-mixed-precision, run only FP32. "
            "If omitted, run both FP32 and BF16 experiments."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32",),
        default="float32",
        help="Parameter dtype for the baseline path. Mixed precision is controlled separately via autocast.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "cs336_systems" / "outputs" / "memory"),
        help="Directory for memory CSV summaries and snapshot files.",
    )
    parser.add_argument(
        "--save-snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a PyTorch memory snapshot pickle for each experiment.",
    )
    parser.add_argument("--snapshot-prefix", default=None, help="Optional prefix to prepend to snapshot filenames.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for model and random token generation.")
    parser.add_argument(
        "--run-all-context-lengths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run context lengths 128, 256, and 512 automatically. Disable to run only --context-length.",
    )
    parser.add_argument("--device", default="cuda", help="CUDA device to run on, e.g. cuda or cuda:0.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="AdamW learning rate for train mode.")
    return parser.parse_args()


def ensure_cuda_is_available(device_name: str) -> None:
    """Exit early with a clear message if CUDA is unavailable."""

    device = torch.device(device_name)
    if device.type != "cuda":
        raise SystemExit("This script requires CUDA. Please pass --device cuda or cuda:<index>.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Memory profiling requires a CUDA-capable machine.")


def determine_context_lengths(args: argparse.Namespace) -> list[int]:
    """Return the context lengths that should be profiled."""

    if args.run_all_context_lengths:
        return DEFAULT_CONTEXT_LENGTHS.copy()
    return [args.context_length]


def determine_modes(args: argparse.Namespace) -> list[str]:
    """Return the benchmark modes to run."""

    if args.mode is None:
        return ["forward", "train"]
    return [args.mode]


def determine_precisions(args: argparse.Namespace) -> list[bool]:
    """Return which precision modes to profile.

    True means BF16 autocast is enabled. False means the model runs in the
    normal FP32 path.
    """

    if args.mixed_precision is None:
        return [False, True]
    return [args.mixed_precision]


def build_config(
    *,
    model_size_key: str,
    context_length: int,
    batch_size: int,
    mode: str,
    mixed_precision: bool,
    args: argparse.Namespace,
) -> BenchmarkConfig:
    """Construct the shared benchmark config used to build the model and data."""

    preset = MODEL_SIZE_PRESETS[model_size_key]
    benchmark_mode = "forward" if mode == "forward" else "train-step"
    return BenchmarkConfig(
        model_size=model_size_key,
        mode=benchmark_mode,
        device=args.device,
        seed=args.seed,
        vocab_size=10_000,
        context_length=context_length,
        batch_size=batch_size,
        rope_theta=10_000.0,
        d_model=preset["d_model"],
        d_ff=preset["d_ff"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        enable_nvtx=False,
        enable_bf16_autocast=mixed_precision,
    )


def is_oom_error(exc: BaseException) -> bool:
    """Return True when the exception corresponds to an out-of-memory failure."""

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(exc).lower()


def format_precision_label(mixed_precision: bool) -> str:
    """Return the precision label used in logs and snapshot filenames."""

    return "bf16" if mixed_precision else "fp32"


def make_snapshot_path(
    *,
    output_dir: Path,
    model_size_key: str,
    context_length: int,
    mode: str,
    mixed_precision: bool,
    snapshot_prefix: str | None,
) -> Path:
    """Build the snapshot path for one experiment."""

    snapshots_dir = output_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    precision = format_precision_label(mixed_precision)
    prefix = "" if not snapshot_prefix else f"{snapshot_prefix}_"
    filename = f"{prefix}{model_size_key}_ctx{context_length}_{mode}_{precision}.pickle"
    return snapshots_dir / filename


def mib_from_bytes(value: int) -> float:
    """Convert bytes to MiB for human-readable reporting."""

    return value / (1024 ** 2)


def reset_cuda_state(device: torch.device) -> None:
    """Clear allocator cache between experiments."""

    synchronize(device)
    torch.cuda.empty_cache()


def run_one_step(
    *,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    config: BenchmarkConfig,
    device: torch.device,
    mode: str,
) -> None:
    """Run one forward or full-train step using the shared BF16 autocast helper."""

    if mode == "forward":
        with torch.no_grad():
            with autocast_context(config, device):
                model(inputs)
        return

    with autocast_context(config, device):
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    loss.backward()
    assert optimizer is not None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def append_csv_row(csv_path: Path, row: dict[str, str | int | float]) -> None:
    """Append one summary row to the peak-memory CSV, creating it if needed."""

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDNAMES)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def print_experiment_start(model_size_key: str, context_length: int, mode: str, mixed_precision: bool) -> None:
    """Print a concise experiment header before the workload starts."""

    print(
        (
            f"[run] model_size={model_size_key} context_length={context_length} "
            f"mode={mode} mixed_precision={mixed_precision}"
        ),
        flush=True,
    )


def print_experiment_end(row: dict[str, str | int | float]) -> None:
    """Print a concise summary after each experiment finishes."""

    if row["status"] != "ok":
        print(
            (
                f"[done] model_size={row['model_size']} context_length={row['context_length']} mode={row['mode']} "
                f"mixed_precision={row['mixed_precision']} status={row['status']} error={row['error_message']}"
            ),
            flush=True,
        )
        return

    summary = (
        f"[done] model_size={row['model_size']} context_length={row['context_length']} mode={row['mode']} "
        f"mixed_precision={row['mixed_precision']} peak_allocated={row['max_memory_allocated_mib']:.2f} MiB "
        f"peak_reserved={row['max_memory_reserved_mib']:.2f} MiB"
    )
    if row["snapshot_path"]:
        summary += f" snapshot={row['snapshot_path']}"
    print(summary, flush=True)


def profile_experiment(
    *,
    model_size_key: str,
    context_length: int,
    mode: str,
    mixed_precision: bool,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, str | int | float]:
    """Run one experiment, optionally dump a memory snapshot, and return one CSV row."""

    config = build_config(
        model_size_key=model_size_key,
        context_length=context_length,
        batch_size=args.batch_size,
        mode=mode,
        mixed_precision=mixed_precision,
        args=args,
    )
    device = validate_runtime(config)

    snapshot_path = ""
    recording_enabled = False

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    row: dict[str, str | int | float] = {
        "model_size": model_size_key,
        "context_length": context_length,
        "batch_size": args.batch_size,
        "mode": mode,
        "mixed_precision": mixed_precision,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "max_memory_allocated_bytes": "",
        "max_memory_allocated_mib": "",
        "max_memory_reserved_bytes": "",
        "max_memory_reserved_mib": "",
        "snapshot_path": "",
        "status": "ok",
        "error_message": "",
    }

    model = None
    optimizer = None
    try:
        model = make_model(config, device)
        model.train(mode == "train")
        inputs, targets = make_batch(config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate) if mode == "train" else None

        for _ in range(args.warmup_steps):
            run_one_step(
                model=model,
                inputs=inputs,
                targets=targets,
                optimizer=optimizer,
                config=config,
                device=device,
                mode=mode,
            )
            synchronize(device)

        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
        recording_enabled = True

        for _ in range(args.measure_steps):
            run_one_step(
                model=model,
                inputs=inputs,
                targets=targets,
                optimizer=optimizer,
                config=config,
                device=device,
                mode=mode,
            )
            synchronize(device)

        max_memory_allocated = torch.cuda.max_memory_allocated(device)
        max_memory_reserved = torch.cuda.max_memory_reserved(device)
        row["max_memory_allocated_bytes"] = max_memory_allocated
        row["max_memory_allocated_mib"] = mib_from_bytes(max_memory_allocated)
        row["max_memory_reserved_bytes"] = max_memory_reserved
        row["max_memory_reserved_mib"] = mib_from_bytes(max_memory_reserved)

        if args.save_snapshot:
            snapshot = make_snapshot_path(
                output_dir=output_dir,
                model_size_key=model_size_key,
                context_length=context_length,
                mode=mode,
                mixed_precision=mixed_precision,
                snapshot_prefix=args.snapshot_prefix,
            )
            torch.cuda.memory._dump_snapshot(str(snapshot))
            snapshot_path = str(snapshot)
            row["snapshot_path"] = snapshot_path

    except Exception as exc:  # noqa: BLE001 - keep the sweep running through OOMs and other failures.
        row["status"] = "oom" if is_oom_error(exc) else "error"
        row["error_message"] = str(exc)
        if args.save_snapshot and recording_enabled:
            try:
                snapshot = make_snapshot_path(
                    output_dir=output_dir,
                    model_size_key=model_size_key,
                    context_length=context_length,
                    mode=mode,
                    mixed_precision=mixed_precision,
                    snapshot_prefix=args.snapshot_prefix,
                )
                torch.cuda.memory._dump_snapshot(str(snapshot))
                snapshot_path = str(snapshot)
                row["snapshot_path"] = snapshot_path
            except Exception as snapshot_exc:  # noqa: BLE001
                if row["error_message"]:
                    row["error_message"] = f"{row['error_message']} | snapshot_error={snapshot_exc}"
                else:
                    row["error_message"] = f"snapshot_error={snapshot_exc}"
    finally:
        if recording_enabled:
            torch.cuda.memory._record_memory_history(enabled=None)
        reset_cuda_state(device)

    return row


def main() -> int:
    """Run memory profiling experiments and save snapshots plus peak-memory CSV rows."""

    args = parse_args()
    ensure_cuda_is_available(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "peak_memory_summary.csv"

    model_size_key = resolve_model_size_name(args.model_size)
    context_lengths = determine_context_lengths(args)
    modes = determine_modes(args)
    precision_settings = determine_precisions(args)

    for context_length in context_lengths:
        for mode in modes:
            for mixed_precision in precision_settings:
                print_experiment_start(model_size_key, context_length, mode, mixed_precision)
                row = profile_experiment(
                    model_size_key=model_size_key,
                    context_length=context_length,
                    mode=mode,
                    mixed_precision=mixed_precision,
                    args=args,
                    output_dir=output_dir,
                )
                append_csv_row(csv_path, row)
                print_experiment_end(row)

    print(f"Peak memory summary CSV: {csv_path}", flush=True)
    if args.save_snapshot:
        print(f"Snapshots directory: {output_dir / 'snapshots'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
