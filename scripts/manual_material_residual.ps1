# material_polar_residual_v1 — 用户手动执行命令（PowerShell / VS Code）
#
# Agent 生成（2026-10-08）。Agent 未执行也不会执行任何 -Mode fit 命令；
# 真实数据拟合由用户手动运行。默认只做只读 dry-run（安全）；
# 显式 -Mode fit 才训练 RC/RCS x 3 折（共 6 次，P0 不重训）。
#
# 使用：& .\scripts\manual_material_residual.ps1            # dry-run
#       & .\scripts\manual_material_residual.ps1 -Mode fit  # 真实训练

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('dryrun', 'fit')]
    [string]$Mode = 'dryrun'
)

conda activate stereo
Set-Location 'D:\Python\Project\my_data'

# ---- 解释器核对（必须精确匹配，否则终止）----
$py = python -c "import sys; print(sys.executable)"
if ($py -ne 'D:\Python\CondaPkgs\stereo\python.exe') {
    Write-Error "interpreter mismatch: $py — aborting (fix conda env first)"
    exit 1
}
Write-Host "interpreter OK: $py"

$Config = 'configs\experiments\material_polar_residual_v1.yaml'
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$RunRoot = "analysis\runs\material_polar_residual_v1\manual_$stamp"

# ---- preflight dry-run（准入；非零必须终止，不得继续 fit）----
$dryArgs = @(
    'scripts\train_material_residual.py',
    '--config', $Config,
    '--output-dir', "$RunRoot\dryrun"
)
& $py @dryArgs --dry-run
if ($LASTEXITCODE -ne 0) {
    Write-Error "preflight dry-run FAILED (exit $LASTEXITCODE) — fitting is not allowed; bring the stderr back for review"
    exit $LASTEXITCODE
}
Write-Host "preflight passed: $RunRoot\dryrun\report.json"

if ($Mode -eq 'dryrun') {
    Write-Host "Mode=dryrun: nothing further runs. For the real experiment re-run with -Mode fit (RC/RCS x 3 folds; P0 is NOT retrained)."
    exit 0
}

# ---- 正式拟合：RC/RCS x 全部 3 折（用户手动执行；CPU，每次约数分钟）----
$fitArgs = @(
    'scripts\train_material_residual.py',
    '--config', $Config,
    '--output-dir', "$RunRoot\fit",
    '--fold', 'all',
    '--variants', 'RC', 'RCS'
)
& $py @fitArgs --fit
if ($LASTEXITCODE -ne 0) {
    Write-Error "fit FAILED (exit $LASTEXITCODE)"
    exit $LASTEXITCODE
}
Write-Host "summary: $RunRoot\fit\summary.json"
Write-Host "shuffle diagnostics: $RunRoot\fit\shuffle\<fold>\"
Write-Host ""
Write-Host "Reading the results:"
Write-Host "  variants.R0/RC/RCS.macro_f1_mean  - three-fold mean (do NOT mix with pooled)"
Write-Host "  variants.R0/RC/RCS.pooled         - pooled outer-eval metrics"
Write-Host "  paired.RCS_minus_R0 / RCS_minus_RC - fixed vs broke counts"
Write-Host "  shuffle diagnostics (per fold):   fit\shuffle\<fold>\seed_<seed>.json"
Write-Host "    coverage / unmatched / same_group_donor_ratio / per_class_changes"
Write-Host ""
Write-Host "Boundaries: group-isolated protocol on the old train+val pool only;"
Write-Host "engineering thresholds, not a significance test; no independent-"
Write-Host "generalization or polarization-benefit claim."
