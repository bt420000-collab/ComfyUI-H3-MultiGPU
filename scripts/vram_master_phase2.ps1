param(
    [string]$PythonExe = "",
    [string]$ComfyRoot = "",
    [string]$Primary = "cuda:0",
    [string]$Secondary = "cuda:1",
    [int]$Sequence = 8192,
    [switch]$TestPinned
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$PluginRoot = Split-Path -Parent $ScriptDir

if ([string]::IsNullOrWhiteSpace($ComfyRoot)) {
    $CustomNodes = Split-Path -Parent $PluginRoot
    $ComfyRoot = Split-Path -Parent $CustomNodes
}
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $SuiteRoot = Split-Path -Parent $ComfyRoot
    $PythonExe = Join-Path $SuiteRoot "python_embeded\python.exe"
}

$PythonExe = [System.IO.Path]::GetFullPath($PythonExe)
$ComfyRoot = [System.IO.Path]::GetFullPath($ComfyRoot)
if (-not (Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (-not (Test-Path $ComfyRoot)) { throw "ComfyUI root not found: $ComfyRoot" }

$env:PYTHONPATH = "$PluginRoot;$ComfyRoot;$env:PYTHONPATH"

Write-Host "=== ComfyUI-H3-VRAM-Master Phase 2: Transport + Exact-SP Exchange ==="
Write-Host "Plugin:  $PluginRoot"
Write-Host "ComfyUI: $ComfyRoot"
Write-Host "Pair:    $Primary <-> $Secondary"
Write-Host ""

Write-Host "[1/2] Pageable host relay: legacy 6-move vs packet 1-move"
& $PythonExe (Join-Path $ScriptDir "relay_packet_benchmark.py") `
    --primary $Primary --secondary $Secondary --sequence $Sequence `
    --transport neutral_pageable --repeats 3
if ($LASTEXITCODE -ne 0) { throw "Pageable relay packet benchmark failed: $LASTEXITCODE" }

if ($TestPinned) {
    Write-Host ""
    Write-Host "[optional] Bounded pinned segmented relay"
    & $PythonExe (Join-Path $ScriptDir "relay_packet_benchmark.py") `
        --primary $Primary --secondary $Secondary --sequence $Sequence `
        --transport neutral_pinned --repeats 3
    if ($LASTEXITCODE -ne 0) { throw "Pinned relay benchmark failed: $LASTEXITCODE" }
}

Write-Host ""
Write-Host "[2/2] Exact-SP exchange numerical parity on Windows-safe pageable relay"
& $PythonExe (Join-Path $ScriptDir "exact_sp_exchange_probe.py") `
    --primary $Primary --secondary $Secondary --sequence 101 `
    --transport neutral_pageable
if ($LASTEXITCODE -ne 0) { throw "Exact-SP exchange probe failed: $LASTEXITCODE" }

Write-Host ""
Write-Host "=== VRAM MASTER PHASE 2 PASS ==="
Write-Host "Packet relay is byte-exact and Exact-SP exchange is numerically exact."
Write-Host "Review the printed packet speedup before enabling more aggressive host relay paths."
