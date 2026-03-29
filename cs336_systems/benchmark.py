from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer
from typing import Iterator

import torch
import torch.nn.functional as F

try:
    from cs336_basics.model import BasicsTransformerLM
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parent.parent
    basics_src = project_root / "cs336-basics"
    if str(basics_src) not in sys.path:
        sys.path.insert(0, str(basics_src))
    from cs336_basics.model import BasicsTransformerLM


MODEL_SIZE_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}

MODEL_SIZE_ALIASES = {
    "small": "small",
    "medium": "medium",
    "large": "large",
    "xl": "xl",
    "2.7b": "2.7b",
    "2.7B": "2.7b",
}


@dataclass
class BenchmarkConfig:
    """Complete configuration for one benchmark run."""

    model_size: str
    mode: str
    device: str
    seed: int
    vocab_size: int
    context_length: int
    batch_size: int
    rope_theta: float
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int
    measure_steps: int
    learning_rate: float
    enable_nvtx: bool
    enable_bf16_autocast: bool


def resolve_model_size_name(model_size: str) -> str:
    """Normalize model-size aliases to the canonical preset key."""

    try:
        return MODEL_SIZE_ALIASES[model_size]
    except KeyError as exc:
        raise ValueError(f"Unknown model size: {model_size}") from exc


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the benchmark script."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark and profile a BasicsTransformerLM forward pass, "
            "forward+backward pass, or full train step."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train-step"),
        default="forward-backward",
        help="Benchmark forward-only, forward+backward, or a full optimizer train step.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZE_ALIASES),
        default="small",
        help="Convenience preset for model hyperparameters from the assignment handout.",
    )
    parser.add_argument("--d-model", type=int, default=None, help="Transformer hidden size.")
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward hidden size.")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of Transformer layers.")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of attention heads.")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size.")
    parser.add_argument("--context-length", type=int, default=128, help="Sequence length.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--rope-theta", type=float, default=10_000.0, help="RoPE theta parameter.")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Number of warmup steps.")
    parser.add_argument("--measure-steps", type=int, default=10, help="Number of timed steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate used by train-step mode.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--enable-nvtx",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Emit NVTX ranges for Nsight Systems profiling.",
    )
    parser.add_argument(
        "--enable-bf16-autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run model forward/loss under torch.autocast(device_type='cuda', dtype=torch.bfloat16).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model/data initialization.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Convert parsed arguments into a structured benchmark config."""

    model_size = resolve_model_size_name(args.model_size)
    preset = MODEL_SIZE_PRESETS[model_size]
    return BenchmarkConfig(
        model_size=model_size,
        mode=args.mode,
        device=args.device,
        seed=args.seed,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        batch_size=args.batch_size,
        rope_theta=args.rope_theta,
        d_model=args.d_model if args.d_model is not None else preset["d_model"],
        d_ff=args.d_ff if args.d_ff is not None else preset["d_ff"],
        num_layers=args.num_layers if args.num_layers is not None else preset["num_layers"],
        num_heads=args.num_heads if args.num_heads is not None else preset["num_heads"],
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        learning_rate=args.learning_rate,
        enable_nvtx=args.enable_nvtx,
        enable_bf16_autocast=args.enable_bf16_autocast,
    )


def synchronize(device: torch.device) -> None:
    """Wait for all queued CUDA work to finish when running on GPU."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def nvtx_range(message: str, *, enabled: bool) -> Iterator[None]:
    """Emit an NVTX range when profiling is enabled."""

    if not enabled:
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def autocast_context(config: BenchmarkConfig, device: torch.device):
    """Return a context manager for optional CUDA BF16 autocast.

    We keep the model parameters in their original dtype and only enable BF16 for
    eligible CUDA ops through autocast. This preserves the FP32 baseline path and
    matches the assignment requirement for mixed precision benchmarking.
    """

    if not config.enable_bf16_autocast:
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("BF16 autocast requires a CUDA device.")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def validate_runtime(config: BenchmarkConfig) -> torch.device:
    """Validate the requested runtime configuration and return the resolved device."""

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available.")
    if config.enable_bf16_autocast:
        if device.type != "cuda":
            raise ValueError("BF16 autocast can only be enabled on CUDA.")
        if hasattr(torch.cuda, "is_bf16_supported") and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA BF16 autocast was requested, but this CUDA device does not report BF16 support.")
    return device


def make_model(config: BenchmarkConfig, device: torch.device) -> BasicsTransformerLM:
    """Create the Transformer model described by the benchmark config."""

    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    )
    return model.to(device)


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random input batch and random token targets."""

    inputs = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(config.batch_size, config.context_length),
        device=device,
    )
    return inputs, targets


def make_optimizer(model: BasicsTransformerLM, config: BenchmarkConfig) -> torch.optim.Optimizer | None:
    """Create the optimizer only for full train-step benchmarking."""

    if config.mode != "train-step":
        return None
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate)


def benchmark_step(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    optimizer: torch.optim.Optimizer | None,
    nvtx_enabled: bool,
    device: torch.device,
) -> None:
    """Run one benchmark step for the selected benchmark mode."""

    if config.mode == "forward":
        with nvtx_range("benchmark.forward", enabled=nvtx_enabled):
            with torch.no_grad():
                with autocast_context(config, device):
                    model(inputs)
        return

    with nvtx_range("benchmark.zero_grad", enabled=nvtx_enabled):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

    with autocast_context(config, device):
        with nvtx_range("benchmark.forward", enabled=nvtx_enabled):
            logits = model(inputs)

        with nvtx_range("benchmark.loss", enabled=nvtx_enabled):
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    with nvtx_range("benchmark.backward", enabled=nvtx_enabled):
        loss.backward()

    if optimizer is not None:
        with nvtx_range("benchmark.optimizer_step", enabled=nvtx_enabled):
            optimizer.step()


def run_benchmark(
    model: BasicsTransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: BenchmarkConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> list[float]:
    """Run warmup plus measured benchmark steps and return per-step timings."""

    nvtx_enabled = config.enable_nvtx and device.type == "cuda"

    with nvtx_range("benchmark.warmup", enabled=nvtx_enabled):
        for _ in range(config.warmup_steps):
            with nvtx_range("benchmark.warmup.step", enabled=nvtx_enabled):
                benchmark_step(model, inputs, targets, config, optimizer, nvtx_enabled, device)
                synchronize(device)

    step_times: list[float] = []
    with nvtx_range("benchmark.measure", enabled=nvtx_enabled):
        for _ in range(config.measure_steps):
            synchronize(device)
            with nvtx_range("benchmark.measure.step", enabled=nvtx_enabled):
                start = default_timer()
                benchmark_step(model, inputs, targets, config, optimizer, nvtx_enabled, device)
                synchronize(device)
                step_times.append(default_timer() - start)
    return step_times


def summarize(step_times: list[float]) -> dict[str, float]:
    """Summarize per-step timings into report-friendly statistics."""

    total_seconds = sum(step_times)
    return {
        "mean_seconds": statistics.mean(step_times),
        "stdev_seconds": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "min_seconds": min(step_times),
        "max_seconds": max(step_times),
        "total_seconds": total_seconds,
        "steps_per_second": len(step_times) / total_seconds if total_seconds > 0 else float("inf"),
    }


def make_csv_row(
    config: BenchmarkConfig,
    summary: dict[str, float],
    step_times: list[float],
) -> dict[str, str | int | float]:
    """Build one CSV record combining config and timing summary."""

    row: dict[str, str | int | float] = {}
    row.update(asdict(config))
    row.update(summary)
    row["step_times_seconds"] = json.dumps(step_times)
    return row


def read_existing_csv_rows(
    csv_path: Path,
    desired_fieldnames: list[str],
) -> tuple[list[dict[str, str]], bool]:
    """Read and normalize rows from an existing benchmark CSV.

    The benchmark CSV schema changed a few times while the assignment notebook
    was being built. This helper keeps old rows readable and rewrites the file
    when the current schema no longer matches the existing header.
    """

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return [], False

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))

    if not rows:
        return [], False

    existing_header = rows[0]
    rewrite_required = existing_header != desired_fieldnames
    normalized_rows: list[dict[str, str]] = []

    for raw_row in rows[1:]:
        if not raw_row:
            continue

        existing_map = dict(zip(existing_header, raw_row))
        looks_like_shifted_new_row = (
            existing_header != desired_fieldnames
            and existing_map.get("stdev_seconds") in {"True", "False"}
        )

        if looks_like_shifted_new_row:
            row_map = dict(zip(desired_fieldnames, raw_row))
            rewrite_required = True
        else:
            row_map = existing_map
            if len(raw_row) != len(existing_header):
                rewrite_required = True

        looks_like_legacy_row_under_new_header = (
            row_map.get("step_times_seconds", "") == ""
            and str(row_map.get("total_seconds", "")).startswith("[")
        )
        if looks_like_legacy_row_under_new_header:
            row_map = {
                **row_map,
                "learning_rate": "",
                "enable_nvtx": "",
                "enable_bf16_autocast": "",
                "mean_seconds": row_map.get("learning_rate", ""),
                "stdev_seconds": row_map.get("enable_nvtx", ""),
                "min_seconds": row_map.get("mean_seconds", ""),
                "max_seconds": row_map.get("stdev_seconds", ""),
                "total_seconds": row_map.get("min_seconds", ""),
                "steps_per_second": row_map.get("max_seconds", ""),
                "step_times_seconds": row_map.get("total_seconds", ""),
            }
            rewrite_required = True

        normalized_rows.append({field: row_map.get(field, "") for field in desired_fieldnames})

    return normalized_rows, rewrite_required


def append_result_to_csv(row: dict[str, str | int | float]) -> Path:
    """Append one benchmark record to results/1.1.3_results.csv."""

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "1.1.3_results.csv"
    desired_fieldnames = list(row.keys())
    existing_rows, rewrite_required = read_existing_csv_rows(csv_path, desired_fieldnames)

    if rewrite_required:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=desired_fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        return csv_path

    should_write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=desired_fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)

    return csv_path


def run_configured_benchmark(config: BenchmarkConfig) -> tuple[dict[str, float], list[float]]:
    """Run one benchmark from a fully constructed config without writing CSV output.

    This helper is shared by the CLI entrypoint and the mixed-precision sweep
    script so both use exactly the same benchmark flow.
    """

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = validate_runtime(config)

    # Keep the model NVTX instrumentation tied to the benchmark-level profiling flag.
    os.environ["CS336_ENABLE_NVTX"] = "1" if config.enable_nvtx and device.type == "cuda" else "0"

    model = make_model(config, device)
    model.train(config.mode != "forward")
    optimizer = make_optimizer(model, config)
    inputs, targets = make_batch(config, device)
    synchronize(device)

    step_times = run_benchmark(model, inputs, targets, config, device, optimizer)
    return summarize(step_times), step_times


def main() -> int:
    """CLI entrypoint."""

    args = parse_args()
    config = build_config(args)

    summary, step_times = run_configured_benchmark(config)
    csv_row = make_csv_row(config, summary, step_times)
    csv_path = append_result_to_csv(csv_row)

    print("Benchmark configuration:")
    print(json.dumps(asdict(config), indent=2))
    print("\nResults:")
    print(json.dumps(summary, indent=2))
    print("\nPer-step timings (seconds):")
    print(json.dumps(step_times))
    print("\nResults CSV:")
    print(str(csv_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
