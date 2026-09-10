param(
    [Parameter(Mandatory=$true)]
    [string]$Model,
    [string]$PythonExe = "",
    [string]$ComfyRoot = "",
    [string]$Device = "cuda:0",
    [int]$Sequence = 64
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

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (-not (Test-Path $ComfyRoot)) {
    throw "ComfyUI root not found: $ComfyRoot"
}
if (-not (Test-Path $Model)) {
    throw "H3 model not found: $Model"
}

Write-Host "=== H3VM Dual16G Phase 0 ==="
Write-Host "Plugin:    $PluginRoot"
Write-Host "ComfyUI:   $ComfyRoot"
Write-Host "Python:    $PythonExe"
Write-Host "Model:     $Model"
Write-Host "Device:    $Device"
Write-Host ""

Write-Host "[1/2] Hardware symmetry probe"
& $PythonExe (Join-Path $ScriptDir "dual16g_lab_probe.py")
if ($LASTEXITCODE -ne 0) {
    throw "Dual16G hardware probe failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "[2/2] Real H3 quantized-QKV parity"
& $PythonExe (Join-Path $ScriptDir "qkv_quantized_parity.py") `
    --comfy-root $ComfyRoot `
    --model $Model `
    --device $Device `
    --sequence $Sequence
if ($LASTEXITCODE -ne 0) {
    throw "Quantized QKV parity failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "=== PHASE 0 PASS ==="
Write-Host "Hardware probe completed and quantized-QKV parity passed."
Write-Host "Next gate: DUAL_QUIET wall-time sweep with H3VM_DUAL16_SYMMETRIC_LAB=1."
