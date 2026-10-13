# material_residual_scale_v1 - manual entry (PowerShell).
#
# Default -Mode dryrun runs the real non-training admission + scale
# computation + 36-path matrix (safe to run any time).
# -Mode fit runs the 36 REAL fits and then the formal independent
# verification with the SAME shared deadline (one UTC instant generated
# here, passed to both phases; the budget is never reset between them).
# Outputs always go to FRESH directories; a timeout keeps the finished
# evidence as PARTIAL and the follow-up verification is skipped.
#
# Native argument passing uses a STRING ARRAY (hashtable splatting is
# not reliable for native executables on Windows PowerShell 5.1); every
# element with spaces stays one argument.
#
# -PythonExe exists ONLY so tests can inject a recording stub
# interpreter; the production default stays the validated project
# interpreter and must not be changed.
#
# Usage:
#   & scripts\manual_material_residual_scale.ps1 -Mode dryrun `
#       -FeatureDir '<cache_build\features absolute path>'
#   & scripts\manual_material_residual_scale.ps1 -Mode fit `
#       -FeatureDir '<cache_build\features absolute path>' `
#       -OutDir '<fresh fit output absolute path>' -BudgetMinutes 60
[CmdletBinding()]
param(
    [ValidateSet('dryrun', 'fit')]
    [string]$Mode = 'dryrun',
    [Parameter(Mandatory = $true)]
    [string]$FeatureDir,
    [string]$OutDir = "",
    [int]$BudgetMinutes = 60,
    [string]$PythonExe = 'D:\Python\CondaPkgs\stereo\python.exe'
)

$ErrorActionPreference = 'Stop'
$TaskRoot = Split-Path -Parent $PSScriptRoot
$TaskConfig = Join-Path $TaskRoot 'configs\experiments\material_residual_scale_v1.yaml'
$TaskP0 = Join-Path $TaskRoot 'analysis\runs\material_existing_data_v1\manual_20261008_085510_822\fit'
$TrainScript = Join-Path $TaskRoot 'scripts\train_material_residual_scale.py'
$VerifyScript = Join-Path $TaskRoot 'scripts\verify_material_residual_scale.py'

if (-not (Test-Path $FeatureDir)) {
    Write-Error "FeatureDir does not exist: $FeatureDir"
    exit 1
}
if ($BudgetMinutes -le 0) {
    Write-Error "BudgetMinutes must be positive"
    exit 1
}
if (-not (Test-Path $PythonExe)) {
    Write-Error "PythonExe does not exist: $PythonExe"
    exit 1
}

# ONE shared UTC deadline for the whole manual run: fit and verify get
# the SAME instant, the budget is never reset between phases. The
# trailing 'Z' is a LITERAL (the custom-format 'zzz' would print the
# machine's local offset while the value is UTC, mislabelling the
# instant by the UTC offset).
$Deadline = (Get-Date).ToUniversalTime().AddMinutes($BudgetMinutes).ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")

if ($Mode -eq 'dryrun') {
    if (-not $OutDir) {
        $stamp = Get-Date -Format 'yyyyMMddTHHmmss'
        $OutDir = Join-Path $TaskRoot ("analysis\runs\material_residual_scale_v1\manual_dryrun_$stamp")
    }
    if (Test-Path $OutDir) {
        Write-Error "OutDir already exists: $OutDir"
        exit 1
    }
    $trainArgs = @(
        $TrainScript,
        '--config', $TaskConfig,
        '--feature-dir', $FeatureDir,
        '--p0-fit-dir', $TaskP0,
        '--output-dir', $OutDir,
        '--dry-run',
        '--deadline-utc', $Deadline
    )
    Write-Host "dry-run -> $OutDir (deadline $Deadline)"
    & $PythonExe @trainArgs
    exit $LASTEXITCODE
}

# -Mode fit: the 36 real fits + the formal independent verification,
# both bounded by the SAME shared deadline
if (-not $OutDir) {
    Write-Error "-Mode fit requires -OutDir (a fresh directory)"
    exit 1
}
if (Test-Path $OutDir) {
    Write-Error "OutDir already exists: $OutDir"
    exit 1
}
$verifyDir = "${OutDir}_verification"
if (Test-Path $verifyDir) {
    Write-Error "verification directory already exists: $verifyDir"
    exit 1
}
$fitArgs = @(
    $TrainScript,
    '--config', $TaskConfig,
    '--feature-dir', $FeatureDir,
    '--p0-fit-dir', $TaskP0,
    '--output-dir', $OutDir,
    '--fit',
    '--deadline-utc', $Deadline
)
Write-Host "fit -> $OutDir (shared deadline $Deadline)"
& $PythonExe @fitArgs
$fitExit = $LASTEXITCODE
if ($fitExit -ne 0) {
    Write-Warning "fit exited with $fitExit; keeping the finished evidence and SKIPPING the verification (a timeout/failure is never reported as a complete success)"
    exit $fitExit
}
$verifyArgs = @(
    $VerifyScript,
    '--config', $TaskConfig,
    '--feature-dir', $FeatureDir,
    '--p0-fit-dir', $TaskP0,
    '--experiment-fit-dir', $OutDir,
    '--output-dir', $verifyDir,
    '--deadline-utc', $Deadline
)
Write-Host "verification -> $verifyDir (same deadline $Deadline)"
& $PythonExe @verifyArgs
exit $LASTEXITCODE
