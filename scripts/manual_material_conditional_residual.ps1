# Manual entry for the conditional (P0-context) residual experiment.
# Default mode is DRY-RUN; training only starts with the explicit -Mode
# fit flag. The agent never runs the real fit.
#
# Usage (PowerShell):
#   & .\scripts\manual_material_conditional_residual.ps1            # dry-run
#   & .\scripts\manual_material_conditional_residual.ps1 -Mode fit  # 18 fits
#
# Results to read after a fit (summary.json):
#   variants.R0 / variants.RC / variants.RCS  (per_seed + mean_of_seeds)
#   paired.RCS_minus_R0 / paired.RCS_minus_RC (per_seed + total)
#   per_class_changes / per_group_changes     (summed_over_seeds)
#   linear_reference                          (the corrected linear fit)

param(
    [ValidateSet("dryrun", "fit")]
    [string]$Mode = "dryrun",
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$py = "D:\Python\CondaPkgs\stereo\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "interpreter not found: $py"
    exit 1
}

Push-Location (Join-Path $PSScriptRoot "..")
try {
    $stamp = Get-Date -Format "yyyyMMddTHHmmss"
    if ($OutDir -eq "") {
        $OutDir = "analysis/runs/material_conditional_residual_v1/manual_$stamp"
    }
    if (Test-Path $OutDir) {
        Write-Error "output directory already exists: $OutDir"
        exit 1
    }

    $baseArgs = @(
        "scripts/train_material_conditional_residual.py",
        "--config", "configs/experiments/material_conditional_residual_v1.yaml",
        "--fit-dir", "analysis/runs/material_existing_data_v1/manual_20261008_085510_822/fit",
        "--output-dir", "$OutDir",
        "--fold", "all"
    )

    if ($Mode -eq "dryrun") {
        $dryArgs = $baseArgs
        & $py @dryArgs --dry-run
    }
    elseif ($Mode -eq "fit") {
        $fitArgs = $baseArgs
        & $py @fitArgs --fit
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Error "entry exited with code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host ""
    Write-Host "Output directory: $OutDir"
    if ($Mode -eq "fit") {
        Write-Host "Read summary.json:"
        Write-Host "  variants.*.per_seed.<seed>.macro_f1_mean   (per-seed three-fold mean)"
        Write-Host "  variants.*.per_seed.<seed>.pooled.macro_f1 (per-seed pooled outer)"
        Write-Host "  variants.*.macro_f1_mean_of_seeds / pooled_macro_f1_mean_of_seeds"
        Write-Host "  paired.*.per_seed / paired.*.total / paired.*.macro_f1_by_variant"
        Write-Host "    (macro_f1_by_variant has fold_mean_of_seeds and pooled_mean_of_seeds per variant)"
        Write-Host "  per_class_changes.* / per_group_changes.* (summed_over_seeds)"
        Write-Host "  linear_reference (bound corrected linear fit identity)"
        Write-Host "Thresholds and the stop condition are pre-registered in the experiment config."
    }
    else {
        Write-Host "Dry-run complete. Run with -Mode fit to start the 18 fits (manual step)."
    }
}
finally {
    Pop-Location
}
