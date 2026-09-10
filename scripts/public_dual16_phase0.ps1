param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [string]$PythonExe = "G:\ComfyUI-MiniMax-H3\python_embeded\python.exe",
    [string]$ComfyRoot = "G:\ComfyUI-MiniMax-H3\ComfyUI",
    [string]$Device = "cuda:0",
    [int]$Sequence = 64
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== H3VM PUBLIC DUAL16 PHASE 0 ==="
Write-Host "Python:  $PythonExe"
Write-Host "ComfyUI: $ComfyRoot"
Write-Host "Model:   $Model"
Write-Host ""

Write-Host "[1/2] Visible GPU probe"
& $PythonExe (Join-Path $ScriptDir "dual_gpu_probe.py")
if ($LASTEXITCODE -ne 0) {
    throw "GPU probe failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[2/2] Real H3 quantized QKV parity"
& $PythonExe (Join-Path $ScriptDir "qkv_quantized_parity.py") `
    --comfy-root $ComfyRoot `
    --model $Model `
    --device $Device `
    --sequence $Sequence `
    --split-head 28
if ($LASTEXITCODE -ne 0) {
    throw "QKV parity failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "=== PHASE 0 PASS ==="
Write-Host "Next: install this branch as the only ComfyUI-H3-MultiGPU copy and run the public Mode4 A/B workload."
Write-Host "Expected Mode4 banner: Dev9.4 25/25 RAM2G PREDICT075."
Write-Host "On Windows expected host policy: windows_multigpu_pageable_guard / pageable / pinned_cap=0MiB."