param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [string]$PythonExe = "",
    [string]$ComfyRoot = "",
    [string]$Primary = "cuda:0",
    [string]$Secondary = "cuda:1",
    [int]$Sequence = 8192,
    [double]$HostGbps = 0.0
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
$Model = [System.IO.Path]::GetFullPath($Model)

if (-not (Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }
if (-not (Test-Path $ComfyRoot)) { throw "ComfyUI root not found: $ComfyRoot" }
if (-not (Test-Path $Model)) { throw "H3 model not found: $Model" }

Write-Host "=== ComfyUI-H3-VRAM-Master Phase 1 ==="
Write-Host "Plugin:  $PluginRoot"
Write-Host "ComfyUI: $ComfyRoot"
Write-Host "Model:   $Model"
Write-Host ""

Write-Host "[1/2] Unified compute planner"
$planArgs = @(
    (Join-Path $ScriptDir "vram_master_plan_probe.py"),
    "--primary", $Primary,
    "--secondary", $Secondary,
    "--sequence", "$Sequence"
)
if ($HostGbps -gt 0) {
    $planArgs += @("--host-gbps", "$HostGbps")
}
& $PythonExe @planArgs
if ($LASTEXITCODE -ne 0) { throw "VRAM Master planner probe failed: $LASTEXITCODE" }

Write-Host ""
Write-Host "[2/2] Real H3 quantized-QKV parity"
& $PythonExe (Join-Path $ScriptDir "qkv_quantized_parity.py") `
    --comfy-root $ComfyRoot `
    --model $Model `
    --device $Primary `
    --sequence 64
if ($LASTEXITCODE -ne 0) { throw "Quantized QKV parity failed: $LASTEXITCODE" }

Write-Host ""
Write-Host "=== VRAM MASTER PHASE 1 PASS ==="
Write-Host "Planner completed and quantized-QKV parity passed."
Write-Host "Next: install this lab branch as ComfyUI-H3-VRAM-Master and run real Capacity/Mode4 A/B."
