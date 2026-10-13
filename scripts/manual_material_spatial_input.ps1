# material_spatial_input_v1 — the manual wrapper (review_fix_v1).
#
# Runs the requested mode end-to-end with ONE shared UTC deadline.
# Admission is never inferred from child exit codes alone: the wrapper
# reads and checks the builder status, the verifier status/report, the
# cache binding and the identity counts before publishing. Only builder
# + verifier + source protection + binding all passing within the
# deadline yields overall_status=COMPLETED with input_admitted=true
# (build mode only; preflight and smoke are never admitted).
#
# Target refusals (existing target, traversal, alias segments, link or
# junction ancestors, outside the experiment root) happen BEFORE any
# creation or write: zero creation, zero overwrite, refusal on stderr.
param(
    [ValidateSet('preflight', 'smoke', 'build')]
    [string]$Mode = 'preflight',
    [Parameter(Mandatory = $true)][string]$OutDir,
    [Parameter(Mandatory = $true)][double]$BudgetMinutes,
    [int]$MaxFrames = 0,
    # test-only injection: the formal default interpreter is fixed below
    [string]$PythonExe = ''
)

$ErrorActionPreference = 'Stop'
# A PS7 parent can pass its module search path to Windows PowerShell.
# Load this host's built-in JSON/hash/date cmdlets without changing any
# global configuration or installing a module.
Import-Module -Name (Join-Path $PSHOME 'Modules\Microsoft.PowerShell.Utility\Microsoft.PowerShell.Utility.psd1') -ErrorAction Stop

# the wrapper's own location defines the project root (the formal script
# lives in <root>\scripts; the sandbox copy in <sandbox>\scripts)
$ProjectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$ExperimentRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $ProjectRoot 'analysis\runs\material_spatial_input_v1'))
$ConfigPath = Join-Path $ProjectRoot 'configs\experiments\material_spatial_input_v1.yaml'
$TaskScript = Join-Path $ProjectRoot 'scripts\prepare_material_spatial_input.py'
$TaskVerify = Join-Path $ProjectRoot 'scripts\verify_material_spatial_input.py'
if ($PythonExe) { $Py = $PythonExe } else {
    $Py = 'D:\Python\CondaPkgs\stereo\python.exe'
}

$script:RunRoot = $null
$script:UtcDeadline = $null
$script:ConfigSha = $null
$script:ClockCalls = 0
$script:PublishRolledBack = $false
$script:UseTestClock = $false

function Get-UtcNow {
    # the formal default clock is the system UTC. Synthetic tests may
    # point MATERIAL_SPATIAL_WRAPPER_CLOCK_FILE at a JSON spec
    # {"base": "<iso>", "step_s": <seconds>}: the k-th call then returns
    # base + (k-1)*step, which makes the deadline boundaries
    # deterministic without touching any other gate.
    $clockFile = $env:MATERIAL_SPATIAL_WRAPPER_CLOCK_FILE
    if ($script:UseTestClock) {
        $spec = Get-Content -LiteralPath $clockFile -Raw |
            ConvertFrom-Json
        $base = [DateTime]::Parse($spec.base).ToUniversalTime()
        $k = $script:ClockCalls
        $script:ClockCalls += 1
        return $base.AddSeconds($k * [double]$spec.step_s)
    }
    return (Get-Date).ToUniversalTime()
}

function Get-UtcStamp {
    # wall-clock timestamps for evidence only; never used for gating
    (Get-Date).ToUniversalTime().ToString("yyyy-MM-dd'T'HH:mm:ss.fff'Z'")
}

function Format-Utc([DateTime]$Value) {
    $Value.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'")
}

function Write-Refusal([string]$Message) {
    [Console]::Error.WriteLine("refused: $Message")
    exit 1
}

function Assert-RunRootTarget([string]$Raw, [string]$Root) {
    if ([string]::IsNullOrWhiteSpace($Raw)) {
        Write-Refusal 'the target is empty'
    }
    if (-not [System.IO.Path]::IsPathRooted($Raw)) {
        Write-Refusal "the target must be an absolute path: $Raw"
    }
    foreach ($Segment in ($Raw -split '[\\/]+')) {
        if ($Segment -eq '..' -or $Segment -eq '.') {
            Write-Refusal "path traversal segment in the target: $Raw"
        }
        if ($Segment -match '~') {
            Write-Refusal "8.3 alias segment in the target: $Raw"
        }
    }
    $full = [System.IO.Path]::GetFullPath($Raw)
    $rootCanon = [System.IO.Path]::GetFullPath($Root)
    $prefix = $rootCanon.TrimEnd('\') + '\'
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        # the experiment root itself, project-internal trees outside it
        # and similar-prefix siblings are all refused alike
        Write-Refusal ("the target $full is not a strict subdirectory " +
            "of the experiment root $rootCanon")
    }
    if (Test-Path -LiteralPath $full) {
        Write-Refusal "the target already exists: $full"
    }
    $probe = $full
    while ($true) {
        $parent = Split-Path -Parent $probe
        if ($null -eq $parent) { break }
        if (Test-Path -LiteralPath $parent) {
            $item = Get-Item -LiteralPath $parent -Force
            if (-not $item.PSIsContainer) {
                Write-Refusal "an ancestor is a file: $parent"
            }
            if (($item.Attributes -band `
                        [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
                Write-Refusal "an ancestor is a link/junction: $parent"
            }
        }
        if ($parent.Equals($rootCanon, [StringComparison]::OrdinalIgnoreCase)) {
            break
        }
        $probe = $parent
    }
    return $full
}

function Read-JsonFile([string]$Path) {
    Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
}

function Assert-CertificateCount($Value, [string]$Field) {
    if (($Value -isnot [int] -and $Value -isnot [long]) -or $Value -lt 0) {
        throw "certificate count $Field must be a nonnegative integer"
    }
}

function Write-JournalEvent([hashtable]$Event) {
    # compressed single-line JSON, APPENDED — prior events survive
    $wrapperDir = Join-Path $script:RunRoot 'wrapper'
    if (-not (Test-Path -LiteralPath $wrapperDir -PathType Container)) {
        New-Item -ItemType Directory -Path $wrapperDir | Out-Null
    }
    $journal = Join-Path $wrapperDir 'publish_history.jsonl'
    if (-not (Test-Path -LiteralPath $journal -PathType Leaf)) {
        [System.IO.File]::WriteAllText($journal, '',
            (New-Object System.Text.UTF8Encoding($false)))
    }
    $Event['at'] = Get-UtcStamp
    $line = ($Event | ConvertTo-Json -Depth 6 -Compress)
    Add-Content -LiteralPath $journal -Value $line -Encoding utf8 `
        -ErrorAction Stop
}

function Publish-Status([string]$Status, [bool]$Admitted, [int]$Code,
                        [string]$Detail, [hashtable]$Extra) {
    # Atomic status publish with the deadline re-checks. Throws on any
    # IO failure; returns after a post-commit rollback when the publish
    # itself crossed the deadline.
    if (-not (Test-Path -LiteralPath $script:RunRoot -PathType Container)) {
        New-Item -ItemType Directory -Path $script:RunRoot | Out-Null
    }
    $destination = Join-Path $script:RunRoot 'run_status.json'
    if (Test-Path -LiteralPath $destination -PathType Container) {
        throw "the status destination is a directory: $destination"
    }
    $payload = [ordered]@{
        experiment      = 'material_spatial_input_v1'
        mode            = $Mode
        run_id          = Split-Path -Leaf $script:RunRoot
        overall_status  = $Status
        input_admitted  = $Admitted
        exit_code       = $Code
        detail          = $Detail
        config_sha256   = $script:ConfigSha
        deadline_utc    = $script:UtcDeadline
        finished_at     = Get-UtcStamp
        clock_source    = 'system'
    }
    if ($script:UseTestClock) {
        $payload.clock_source = 'test_sequence'
    }
    foreach ($key in $Extra.Keys) { $payload[$key] = $Extra[$key] }
    $json = $payload | ConvertTo-Json -Depth 6
    $tmp = Join-Path $script:RunRoot 'run_status.json.tmp'
    [System.IO.File]::WriteAllText($tmp, $json,
        (New-Object System.Text.UTF8Encoding($false)))
    # The effective result, not the caller's requested result, controls
    # both the published status and the native exit code for EVERY mode.
    if ((Get-UtcNow) -ge `
            [DateTime]::Parse($script:UtcDeadline).ToUniversalTime()) {
        if ($Status -eq 'COMPLETED') {
            $Status = 'PARTIAL'; $Admitted = $false; $Code = 2
            $Detail = "$Detail (the deadline was crossed before the commit)"
            $payload.overall_status = $Status
            $payload.input_admitted = $Admitted
            $payload.exit_code = $Code
            $payload.detail = $Detail
            $json = $payload | ConvertTo-Json -Depth 6
            [System.IO.File]::WriteAllText($tmp, $json,
                (New-Object System.Text.UTF8Encoding($false)))
        }
    }
    Move-Item -LiteralPath $tmp -Destination $destination -Force
    Write-JournalEvent @{ published = $Status; admitted = $Admitted
                          code = $Code; detail = $Detail }
    # post-commit deadline check: an overrun after a COMPLETED commit is
    # rolled back to PARTIAL immediately
    if ((Get-UtcNow) -ge `
            [DateTime]::Parse($script:UtcDeadline).ToUniversalTime()) {
        if ($Status -eq 'COMPLETED') {
            $Status = 'PARTIAL'; $Admitted = $false; $Code = 2
            $Detail = 'the deadline was crossed during the final publish; the committed COMPLETED status was rolled back'
            $payload.overall_status = $Status
            $payload.input_admitted = $Admitted
            $payload.exit_code = $Code
            $payload.detail = $Detail
            $payload.finished_at = Get-UtcStamp
            $rollbackJson = $payload | ConvertTo-Json -Depth 6
            $tmp2 = Join-Path $script:RunRoot 'run_status.json.tmp'
            [System.IO.File]::WriteAllText($tmp2, $rollbackJson,
                (New-Object System.Text.UTF8Encoding($false)))
            Move-Item -LiteralPath $tmp2 -Destination $destination -Force
            Write-JournalEvent @{ published = 'PARTIAL'; admitted = $false
                                  code = 2
                                  detail = 'rollback of the crossed publish' }
            $script:PublishRolledBack = $true
        }
    }
    return [PSCustomObject]@{ status = $Status; admitted = $Admitted
                             code = $Code; detail = $Detail }
}

function Publish-FailureClosure([string]$Detail, [hashtable]$Extra) {
    # unified failure closure: the final run_status must never keep
    # true/exit0 after a failure, and a failed journal must not block
    # the failure status itself
    $destination = Join-Path $script:RunRoot 'run_status.json'
    try {
        Publish-Status 'FAILED' $false 1 $Detail $Extra
    } catch {
        [Console]::Error.WriteLine(
            "publish-failure closure could not write the failure status: $_")
        if (Test-Path -LiteralPath $destination) {
            try {
                Remove-Item -LiteralPath $destination -Force `
                    -ErrorAction Stop
                [Console]::Error.WriteLine(
                    "the previously committed success status was removed: $destination")
            } catch {
                [Console]::Error.WriteLine(
                    "the status at $destination could not be invalidated: $_")
            }
        }
        [Console]::Error.WriteLine("failure closure detail: $Detail")
    }
}

# --- mode/parameter gates: nothing is created before they pass --------
$clockFile = $env:MATERIAL_SPATIAL_WRAPPER_CLOCK_FILE
if ($clockFile) {
    # A test clock is available only to an explicitly injected FakePython
    # inside an owned .test_artifacts sandbox. The formal interpreter
    # always refuses the test variable before any process or write.
    $sandboxAncestor = Get-Item -LiteralPath $ProjectRoot -Force
    while ($null -ne $sandboxAncestor -and $sandboxAncestor.Name -ne '.test_artifacts') {
        $sandboxAncestor = $sandboxAncestor.Parent
    }
    if (-not $PythonExe -or [System.IO.Path]::GetFileName($PythonExe) -ne 'FakePython.cmd' -or
            $null -eq $sandboxAncestor -or
            -not (Test-Path -LiteralPath $clockFile -PathType Leaf)) {
        Write-Refusal 'the test clock requires an explicit FakePython scratch sandbox'
    }
    $script:UseTestClock = $true
}
if ([double]::IsNaN($BudgetMinutes) -or [double]::IsInfinity($BudgetMinutes) `
        -or $BudgetMinutes -le 0) {
    Write-Refusal 'BudgetMinutes must be finite and > 0'
}
$budgetCap = 20.0
if ($Mode -eq 'preflight' -or $Mode -eq 'smoke') { $budgetCap = 5.0 }
if ($BudgetMinutes -gt $budgetCap) {
    Write-Refusal "BudgetMinutes $BudgetMinutes exceeds the $Mode cap of $budgetCap"
}
if ($Mode -eq 'smoke') {
    if ($MaxFrames -lt 4 -or $MaxFrames -gt 12) {
        Write-Refusal 'smoke requires MaxFrames within 4..12 (four empty frames are mandatory)'
    }
} elseif ($MaxFrames -ne 0) {
    Write-Refusal "MaxFrames is only valid for smoke"
}
if (-not (Test-Path -LiteralPath $Py -PathType Leaf)) {
    Write-Refusal "the python executable does not exist: $Py"
}
if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    Write-Refusal "the batch config is missing: $ConfigPath"
}

# --- the target is validated before ANY creation or write -------------
$script:RunRoot = Assert-RunRootTarget $OutDir $ExperimentRoot
$script:ConfigSha = Get-Sha256 $ConfigPath
$script:UtcDeadline = Format-Utc (Get-UtcNow).AddMinutes($BudgetMinutes)
Write-Output "mode=$Mode deadline=$($script:UtcDeadline) outDir=$script:RunRoot"

$builderStatus = 'NOT_RUN'
$verifierStatus = 'NOT_RUN'
$checks = [ordered]@{}
try {
    # --- builder phase -------------------------------------------------
    $prepArgs = @('-B', $TaskScript,
        '--config', $ConfigPath,
        '--mode', $Mode,
        '--output-dir', $script:RunRoot,
        '--deadline-utc', $script:UtcDeadline)
    if ($Mode -eq 'smoke') {
        $prepArgs += @('--max-frames', "$MaxFrames")
    }
    $ErrorActionPreference = 'Continue'
    & $Py @prepArgs
    $prepCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($null -eq $prepCode) {
        throw 'the builder did not run as a native process'
    }
    if ($prepCode -eq 0) {
        $builderStatus = if ($Mode -eq 'preflight') { 'COMPLETED' }
                         else { 'BUILT_UNVERIFIED' }
    }
    elseif ($prepCode -eq 2) { $builderStatus = 'PARTIAL(2)' }
    else { $builderStatus = "FAILED($prepCode)" }
    Write-Output "builder exit: $prepCode"
    if ($prepCode -eq 2) {
        # a PARTIAL builder keeps PARTIAL/false/exit2 and never reaches
        # the verifier
        Publish-Status 'PARTIAL' $false 2 'builder ended PARTIAL' `
            @{builder_status = $builderStatus
              verifier_status = 'NOT_RUN'}
        Write-Output 'overall: PARTIAL (builder exit 2)'
        exit 2
    }
    if ($prepCode -ne 0) {
        Publish-Status 'FAILED' $false $prepCode 'builder failed' `
            @{builder_status = $builderStatus
              verifier_status = 'NOT_RUN'}
        exit $prepCode
    }

    # --- the builder evidence gate: exit0 is only necessary ------------
    $builderEvidencePath = Join-Path $script:RunRoot 'builder\run_status.json'
    if (-not (Test-Path -LiteralPath $builderEvidencePath -PathType Leaf)) {
        Publish-Status 'FAILED' $false 1 'the builder status is missing' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    $b = Read-JsonFile $builderEvidencePath
    $expectBuilderOverall = 'BUILT_UNVERIFIED'
    if ($Mode -eq 'preflight') { $expectBuilderOverall = 'COMPLETED' }
    $checks.builder_overall = "$($b.overall_status) == $expectBuilderOverall"
    if ("$($b.overall_status)" -ne $expectBuilderOverall) {
        Publish-Status 'FAILED' $false 1 'the builder status does not attest the build' `
            @{builder_status = "$($b.overall_status)"
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ("$($b.experiment)" -ne 'material_spatial_input_v1' -or `
            "$($b.run_mode)" -ne $Mode -or `
            "$($b.run_id)" -ne (Split-Path -Leaf $script:RunRoot)) {
        Publish-Status 'FAILED' $false 1 'builder identity mismatch (experiment/run_mode/run_id)' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ("$($b.config_sha256)" -ne $script:ConfigSha -or `
            "$($b.deadline_utc)" -ne $script:UtcDeadline) {
        Publish-Status 'FAILED' $false 1 'builder identity mismatch (config/deadline)' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ("$($b.cache_schema)" -ne 'material_spatial_input_cache_v1' -or `
            "$($b.implementation_revision)" -ne 'review_fix_v1') {
        Publish-Status 'FAILED' $false 1 'the builder cache contract markers are missing' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ("$($b.exit_code)" -ne '0' -or "$($b.failure)" -ne '' -or `
            "$($b.interrupted_by_deadline)" -eq 'True') {
        Publish-Status 'FAILED' $false 1 'the builder status carries a failure' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ($b.unfinished -and $b.unfinished.Count -gt 0) {
        Publish-Status 'FAILED' $false 1 'the builder left unfinished stages' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    $requiredStages = @('admission', 'protection')
    if ($Mode -ne 'preflight') { $requiredStages += 'build' }
    if ($Mode -eq 'build') { $requiredStages += 'disk_space' }
    foreach ($stageName in $requiredStages) {
        $entry = $b.stages.$stageName
        if ($null -eq $entry -or "$($entry.status)" -ne 'COMPLETED') {
            Publish-Status 'FAILED' $false 1 "the builder stage $stageName did not complete" `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
    }
    if ($Mode -eq 'build') {
        $disk = $b.stages.disk_space
        Assert-CertificateCount $disk.free_bytes 'disk_space.free_bytes'
        Assert-CertificateCount $disk.required_bytes 'disk_space.required_bytes'
        if ($disk.required_bytes -ne 10737418240 -or
                $disk.free_bytes -lt $disk.required_bytes -or
                "$($disk.path)" -ne $script:RunRoot) {
            throw 'the builder disk-space certificate does not satisfy the frozen target-volume requirement'
        }
        $checks.disk_space = "$($disk.free_bytes) free bytes before build; $($disk.required_bytes) required"
    }
    if ("$($b.input_admitted)" -ne 'False') {
        Publish-Status 'FAILED' $false 1 'the builder must never self-admit' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ($Mode -eq 'build' -and "$($b.cache_complete)" -ne 'True') {
        Publish-Status 'FAILED' $false 1 'the full cache is not marked complete' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ($Mode -eq 'preflight') {
        if ("$($b.cache_kind)" -ne '' -and "$($b.cache_kind)" -ne 'None') {
            Publish-Status 'FAILED' $false 1 'preflight must not carry a cache kind' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
    }
    # protection: identical must be a real true with no problems, and the
    # independent source manifest must exist and agree
    $protection = $b.protection
    if ($null -eq $protection -or `
            "$($protection.identical)" -ne 'True' -or `
            $protection.identical -isnot [bool]) {
        Publish-Status 'FAILED' $false 1 'the builder protection did not pass as a real bool' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    if ($protection.problems -and $protection.problems.Count -gt 0) {
        Publish-Status 'FAILED' $false 1 'the builder protection reported problems' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    $builderManifestPath = Join-Path $script:RunRoot `
        'builder\source_manifest.json'
    if (-not (Test-Path -LiteralPath $builderManifestPath -PathType Leaf)) {
        Publish-Status 'FAILED' $false 1 "the builder source manifest is missing" `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    $builderManifest = Read-JsonFile $builderManifestPath
    if ("$($builderManifest.identical)" -ne 'True' -or `
            $builderManifest.items.Count -ne $protection.registered_sources) {
        Publish-Status 'FAILED' $false 1 'the builder source manifest disagrees with the protection summary' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        exit 1
    }
    $checks.builder_protection = "identical, $($protection.registered_sources) sources"

    # --- the cache binding: freeze the confirmed digest ----------------
    if ($Mode -ne 'preflight') {
        Assert-CertificateCount $b.built_frames 'builder.built_frames'
        Assert-CertificateCount $b.built_instances 'builder.built_instances'
        $bindingPath = Join-Path $script:RunRoot 'cache\binding.json'
        if (-not (Test-Path -LiteralPath $bindingPath -PathType Leaf)) {
            Publish-Status 'FAILED' $false 1 'the cache binding is missing' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        $bindingSha = Get-Sha256 $bindingPath
        if ("$($b.cache_binding_sha256)" -ne $bindingSha) {
            Publish-Status 'FAILED' $false 1 'the builder status does not bind the actual cache binding' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        $checks.builder_binding = "builder == actual ($($bindingSha.Substring(0, 12))...)"

        # --- verifier phase (never entered after any builder refusal) --
        # the binding must still match right before the verifier starts
        if ((Get-Sha256 $bindingPath) -ne $bindingSha) {
            Publish-Status 'FAILED' $false 1 'the cache binding drifted before the verifier started' `
                @{builder_status = $builderStatus
                  verifier_status = 'NOT_RUN'; checks = $checks}
            exit 1
        }
        $verifyMode = 'smoke'
        if ($Mode -eq 'build') { $verifyMode = 'full' }
        $verifyArgs = @('-B', $TaskVerify,
            '--config', $ConfigPath,
            '--mode', $verifyMode,
            '--cache-dir', $script:RunRoot,
            '--output-dir', (Join-Path $script:RunRoot 'verification'),
            '--deadline-utc', $script:UtcDeadline,
            '--expected-binding-sha256', $bindingSha)
        $ErrorActionPreference = 'Continue'
        & $Py @verifyArgs
        $verifyCode = $LASTEXITCODE
        $ErrorActionPreference = 'Stop'
        if ($null -eq $verifyCode) {
            throw 'the verifier did not run as a native process'
        }
        $verifierStatus = if ($verifyCode -eq 0) { 'VERIFIED' }
                          else { "FAILED($verifyCode)" }
        Write-Output "verifier exit: $verifyCode"
        if ($verifyCode -eq 2) {
            Publish-Status 'PARTIAL' $false 2 'verifier ended PARTIAL' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            Write-Output 'overall: PARTIAL (verifier exit 2)'
            exit 2
        }
        if ($verifyCode -ne 0) {
            Publish-Status 'FAILED' $false $verifyCode 'verifier failed' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit $verifyCode
        }

        # --- the verifier evidence gate --------------------------------
        $verifyStatusPath = Join-Path $script:RunRoot `
            'verification\status.json'
        $verifyReportPath = Join-Path $script:RunRoot `
            'verification\report.json'
        if (-not (Test-Path -LiteralPath $verifyStatusPath -PathType Leaf) -or `
                -not (Test-Path -LiteralPath $verifyReportPath -PathType Leaf)) {
            Publish-Status 'FAILED' $false 1 'the verifier evidence is missing' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        $v = Read-JsonFile $verifyStatusPath
        if ("$($v.overall_status)" -ne 'VERIFIED' -or `
                "$($v.exit_code)" -ne '0' -or `
                "$($v.mode)" -ne $verifyMode -or `
                "$($v.config_sha256)" -ne $script:ConfigSha -or `
                "$($v.deadline_utc)" -ne $script:UtcDeadline -or `
                "$($v.run_id)" -ne (Split-Path -Leaf $script:RunRoot) -or `
                "$($v.cache_binding_sha256)" -ne $bindingSha -or `
                "$($v.cache_schema)" -ne 'material_spatial_input_cache_v1' -or `
                "$($v.implementation_revision)" -ne 'review_fix_v1') {
            Publish-Status 'FAILED' $false 1 'verifier identity/contract mismatch' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        if ("$($v.failure)" -ne '' -or `
                ($v.unfinished -and $v.unfinished.Count -gt 0)) {
            Publish-Status 'FAILED' $false 1 'the verifier status carries a failure' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        foreach ($stageName in @('inventory', 'cache_identity',
                'independent_rebuild', 'protection', 'final_binding')) {
            $entry = $v.stages.$stageName
            if ($null -eq $entry -or "$($entry.status)" -ne 'COMPLETED') {
                Publish-Status 'FAILED' $false 1 "the verifier stage $stageName did not complete" `
                    @{builder_status = $builderStatus
                      verifier_status = $verifierStatus; checks = $checks}
                exit 1
            }
        }
        if ("$($v.protection.identical)" -ne 'True' -or `
                ($v.protection.problems -and $v.protection.problems.Count -gt 0)) {
            Publish-Status 'FAILED' $false 1 'the verifier source protection did not pass' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        if ("$($v.stages.final_binding.sha256)" -ne $bindingSha) {
            Publish-Status 'FAILED' $false 1 'the final binding digest drifted during consumption' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        $verifyReport = Read-JsonFile $verifyReportPath
        if ("$($v.report_sha256)" -ne (Get-Sha256 $verifyReportPath)) {
            throw 'the verifier report does not match its status-bound digest'
        }
        foreach ($field in @('experiment', 'run_id', 'mode', 'config_sha256',
                'deadline_utc', 'cache_binding_sha256', 'cache_schema',
                'implementation_revision')) {
            if ($null -eq $verifyReport.$field -or
                    "$($verifyReport.$field)" -ne "$($v.$field)") {
                throw "verifier report identity mismatch: $field"
            }
        }
        if ($null -ne $verifyReport.failure -or $null -eq $verifyReport.problems -or
                @($verifyReport.problems).Count -ne 0) {
            throw 'the verifier report carries a failure or problems'
        }
        foreach ($field in @('rebuilt_frames', 'rebuilt_instances', 'ledger_rows')) {
            $value = $verifyReport.counts.$field
            Assert-CertificateCount $value "report.counts.$field"
            Assert-CertificateCount $v.outcome.$field "status.outcome.$field"
            Assert-CertificateCount $v.stages.independent_rebuild.$field "stage.$field"
            Assert-CertificateCount $verifyReport.outcome.$field "report.outcome.$field"
            if (
                    $value -ne $v.outcome.$field -or
                    $value -ne $v.stages.independent_rebuild.$field -or
                    $value -ne $verifyReport.outcome.$field) {
                throw "verifier report/status/stage count mismatch: $field"
            }
        }
        $builderFrames = "$($b.built_frames)"
        $builderInstances = "$($b.built_instances)"
        $verifyFrames = "$($verifyReport.counts.rebuilt_frames)"
        $verifyInstances = "$($verifyReport.counts.rebuilt_instances)"
        if ($builderFrames -ne $verifyFrames -or `
                $builderInstances -ne $verifyInstances -or `
                $builderFrames -eq '' -or $builderInstances -eq '') {
            Publish-Status 'FAILED' $false 1 'builder and verifier disagree on the identity counts' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
        $checks.identity_counts = "builder $builderFrames/$builderInstances == verifier $verifyFrames/$verifyInstances"
        if ($Mode -eq 'build') {
            # the formal build must equal the APPROVED dev pool, not just
            # two mutually consistent numbers
            $definition = Read-JsonFile `
                (Join-Path $script:RunRoot 'cache\definition.json')
            $pool = $definition.dev_pool
            $built = $definition.built
            if ("$($pool.nonempty_frames)" -ne '941' -or `
                    "$($pool.instances)" -ne '2446' -or `
                    "$($pool.empty_frames)" -ne '4' -or `
                    "$($built.frames)" -ne '941' -or `
                    "$($built.instances)" -ne '2446' -or `
                    $builderFrames -ne '941' -or $verifyFrames -ne '941' -or `
                    $builderInstances -ne '2446' -or $verifyInstances -ne '2446' -or `
                    "$($verifyReport.counts.ledger_rows)" -ne '2446') {
                Publish-Status 'FAILED' $false 1 'the formal build does not match the approved 941/2446/4 dev pool' `
                    @{builder_status = $builderStatus
                      verifier_status = $verifierStatus; checks = $checks}
                exit 1
            }
            $checks.approved_counts = '941 non-empty frames / 2446 instances / 4 empty frames'
        }
        $checks.verifier_protection = 'identical'

        # the binding and the builder evidence must still match right
        # before the final publish (consumption-period drift)
        if ((Get-Sha256 $bindingPath) -ne $bindingSha) {
            Publish-Status 'FAILED' $false 1 'the cache binding drifted before the final publish' `
                @{builder_status = $builderStatus
                  verifier_status = $verifierStatus; checks = $checks}
            exit 1
        }
    }

    # --- the final publish is a hard deadline boundary ------------------
    if ((Get-UtcNow) -ge `
            [DateTime]::Parse($script:UtcDeadline).ToUniversalTime()) {
        # already expired before the publish: PARTIAL, never a
        # COMPLETED/true write first
        Publish-Status 'PARTIAL' $false 2 `
            'the deadline was crossed before the final publish' `
            @{builder_status = $builderStatus
              verifier_status = $verifierStatus; checks = $checks}
        Write-Output 'overall: PARTIAL (expired before the publish)'
        exit 2
    }
    $admitted = ($Mode -eq 'build')
    $successDetail = if ($Mode -eq 'preflight') {
            'preflight admission, census, full V5 gate and source protection completed; verifier and cache binding not run'
        } elseif ($Mode -eq 'smoke') {
            'smoke builder and verifier passed; smoke caches are never admitted'
        } else {
            'builder, verifier, source protection and binding all passed within the deadline'
        }
    $published = Publish-Status 'COMPLETED' $admitted 0 $successDetail `
        @{builder_status = $builderStatus; verifier_status = $verifierStatus
          checks = $checks}
    if ($published.code -ne 0) {
        Write-Output 'overall: PARTIAL (the publish crossed the deadline; rolled back)'
        exit $published.code
    }
    if ($published.admitted) {
        Write-Output 'overall: COMPLETED (input_admitted=true)'
    } else {
        Write-Output 'overall: COMPLETED (input_admitted=false)'
    }
    exit 0
} catch {
    # unified failure closure: never leave a success status standing
    $detail = "$_"
    [Console]::Error.WriteLine("wrapper failure: $detail")
    Publish-FailureClosure $detail `
        @{builder_status = $builderStatus; verifier_status = $verifierStatus}
    exit 1
}
