from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer
from typing import Iterator

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM


MODEL_SIZE_PRESETS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7b": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


@dataclass
class BenchmarkConfig:
    """保存一次 benchmark 运行所需的完整配置。"""

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


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    这份脚本既服务于 1.1.3 的基础计时，也服务于 1.1.4 的 Nsight 分析，
    因此参数里除了模型超参数和 benchmark 控制项外，还包含：
    - `train-step` 模式：用于 profile 完整训练一步。
    - `--enable-nvtx`：用于在时间线上打 NVTX 标注，便于过滤 warmup
      和观察 forward / backward / optimizer step 的边界。
    """

    parser = argparse.ArgumentParser(
        description="Benchmark and profile a BasicsTransformerLM forward pass, backward pass, or full train step.",
    )
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train-step"),
        default="forward-backward",
        help="Benchmark forward-only, forward+backward, or a full optimizer train step.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZE_PRESETS),
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for model/data initialization.")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    """把命令行参数整理成结构化的 benchmark 配置。"""

    preset = MODEL_SIZE_PRESETS[args.model_size]
    return BenchmarkConfig(
        model_size=args.model_size,
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
    )


def synchronize(device: torch.device) -> None:
    """在 CUDA 设备上等待异步计算完成。"""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def nvtx_range(message: str, *, enabled: bool) -> Iterator[None]:
    """在需要时发出一段 NVTX range。"""

    if not enabled:
        yield
        return

    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def make_model(config: BenchmarkConfig, device: torch.device) -> BasicsTransformerLM:
    """根据配置创建基础 Transformer 并移动到目标设备。"""

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
    """生成一批随机输入 token 和随机目标 token。"""

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


def make_optimizer(
    model: BasicsTransformerLM,
    config: BenchmarkConfig,
) -> torch.optim.Optimizer | None:
    """按模式创建优化器。"""

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
) -> None:
    """执行一次 benchmark step。"""

    if config.mode == "forward":
        with nvtx_range("benchmark.forward", enabled=nvtx_enabled):
            with torch.no_grad():
                model(inputs)
        return

    with nvtx_range("benchmark.zero_grad", enabled=nvtx_enabled):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)

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
    """执行完整 benchmark，包括 warmup 和正式计时。"""

    nvtx_enabled = config.enable_nvtx and device.type == "cuda"

    with nvtx_range("benchmark.warmup", enabled=nvtx_enabled):
        for _ in range(config.warmup_steps):
            with nvtx_range("benchmark.warmup.step", enabled=nvtx_enabled):
                benchmark_step(model, inputs, targets, config, optimizer, nvtx_enabled)
                synchronize(device)

    step_times = []
    with nvtx_range("benchmark.measure", enabled=nvtx_enabled):
        for _ in range(config.measure_steps):
            synchronize(device)
            with nvtx_range("benchmark.measure.step", enabled=nvtx_enabled):
                start = default_timer()
                benchmark_step(model, inputs, targets, config, optimizer, nvtx_enabled)
                synchronize(device)
                step_times.append(default_timer() - start)
    return step_times


def summarize(step_times: list[float]) -> dict[str, float]:
    """把逐步耗时汇总成便于写报告的统计量。"""

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
    """构造一条可直接写入 CSV 的记录。"""

    row: dict[str, str | int | float] = {}
    row.update(asdict(config))
    row.update(summary)
    row["step_times_seconds"] = json.dumps(step_times)
    return row


def read_existing_csv_rows(
    csv_path: Path,
    desired_fieldnames: list[str],
) -> tuple[list[dict[str, str]], bool]:
    """???????? CSV ???

    ????????????????????????????????????????
    ???????
    - ???????????????????????????????
    - ??????????????????????????????
    - ???????????????????????????????????????
    - ????????????????????????????????
    - ????????????????????????????????????
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
            # ?????????????????????????????
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
    """把一次 benchmark 的配置与结果追加写入 CSV。

    写入位置仍然沿用 1.1.3 已经约定好的
    `<project_root>/results/1.1.3_results.csv`，这样前后实验数据仍然集中在
    同一个结果文件里，便于后续整理。
    """

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


def main() -> int:
    """脚本主入口。"""

    args = parse_args()
    config = build_config(args)

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available.")

    # 让基础模型内部的 attention 也能根据同一个开关发出 NVTX ranges。
    os.environ["CS336_ENABLE_NVTX"] = "1" if config.enable_nvtx and device.type == "cuda" else "0"

    model = make_model(config, device)
    model.train(config.mode != "forward")
    optimizer = make_optimizer(model, config)
    inputs, targets = make_batch(config, device)
    synchronize(device)

    step_times = run_benchmark(model, inputs, targets, config, device, optimizer)
    summary = summarize(step_times)
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
