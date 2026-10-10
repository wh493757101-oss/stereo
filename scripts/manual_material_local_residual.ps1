# Manual entry for the local interior/band residual experiment
# (material_polar_local_interior_band_v1). Default mode is DRY-RUN;
# training only starts with the explicit -Mode fit flag. The agent never
# runs the real fit.
#
# Usage (PowerShell):
#   & .\scripts\manual_material_local_residual.ps1            # dry-run
#   & .\scripts\manual_material_local_residual.ps1 -Mode fit  # real fit
#
# Results to read after a fit (summary.json):
#   variants.R0 / variants.RC / variants.RCS  (macro_f1_mean, pooled)
#   paired.RCS_minus_R0 / paired.RCS_minus_RC (fixed/broke)
#   old_global_reference.variants             (the reviewed global run)
#   shuffle.<fold>.seed_<seed>.json           (per-sample evidence)

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
        $OutDir = "analysis/runs/material_polar_local_interior_band_v1/manual_$stamp"
    }
    if (Test-Path $OutDir) {
        Write-Error "output directory already exists: $OutDir"
        exit 1
    }

    $baseArgs = @(
        "scripts/train_material_local_residual.py",
        "--config", "configs/experiments/material_polar_local_interior_band_v1.yaml",
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
        Write-Host "Read summary.json: variants.* / paired.* / old_global_reference.* / shuffle.*"
        Write-Host "Engineering thresholds are listed in the experiment protocol; do not retune."
    }
    else {
        Write-Host "Dry-run complete. Run with -Mode fit to start the real training (manual step)."
    }
}
finally {
    Pop-Location
}
