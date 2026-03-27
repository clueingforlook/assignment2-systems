param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# 固定本小节要求的模型规模与 warmup 组合，总共 3 x 3 = 9 次实验。
$modelSizes = @("small", "medium", "large")
$warmupStepsList = @(0, 1, 5)

# Nsight 的输出不再落到仓库 results\sight，而是放到系统临时目录。
$nsysOutputDir = Join-Path $env:TEMP "cs336_1.1.3_nsys"
if (-not $DryRun) {
    New-Item -ItemType Directory -Force $nsysOutputDir | Out-Null
}

# 这部分 benchmark 参数在 9 次实验中保持不变。
$benchmarkArgs = @(
    "run",
    "python",
    "-m",
    "cs336_systems.benchmark",
    "--context-length", "128",
    "--batch-size", "4",
    "--measure-steps", "10",
    "--mode", "forward",
    "--enable-nvtx"
)

$totalRuns = $modelSizes.Count * $warmupStepsList.Count
$currentRun = 0

foreach ($modelSize in $modelSizes) {
    foreach ($warmupSteps in $warmupStepsList) {
        $currentRun += 1

        $currentBenchmarkArgs = $benchmarkArgs + @(
            "--model-size", $modelSize,
            "--warmup-steps", $warmupSteps
        )

        $nsysOutputPrefix = Join-Path $nsysOutputDir ("1.1.3_{0}_warmup{1}" -f $modelSize, $warmupSteps)
        $commandArgs = @(
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o", $nsysOutputPrefix,
            "uv"
        ) + $currentBenchmarkArgs

        Write-Host ""
        Write-Host "[$currentRun/$totalRuns] 正在运行实验：model_size=$modelSize, warmup_steps=$warmupSteps"
        Write-Host ("nsys " + ($commandArgs -join " "))
        Write-Host "benchmark 结果会继续追加到 results\\1.1.3_results.csv"

        if ($DryRun) {
            continue
        }

        & nsys @commandArgs

        if ($LASTEXITCODE -ne 0) {
            throw "实验失败：model_size=$modelSize, warmup_steps=$warmupSteps"
        }
    }
}

Write-Host ""
Write-Host "9 次实验已完成。benchmark 结果已追加到 results\\1.1.3_results.csv。"
Write-Host ("Nsight 原始报告输出目录：" + $nsysOutputDir)
