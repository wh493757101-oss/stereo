Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function New-WorkerOutcome {
    param(
        [Parameter(Mandatory = $true)]
        [System.Collections.IDictionary]$Payload,
        [Parameter(Mandatory = $true)]
        [int]$ExitCode
    )

    [pscustomobject]@{
        ExitCode = $ExitCode
        Payload  = $Payload
    }
}

function New-RejectedOutcome {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    New-WorkerOutcome -ExitCode 2 -Payload ([ordered]@{
        status  = "rejected"
        message = $Message
    })
}

function Test-ScopedBashTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Tool
    )

    if ($Tool.Contains("`r") -or $Tool.Contains("`n")) {
        return $false
    }
    if ($Tool -notmatch '^Bash\((?<command>.+)\)$') {
        return $false
    }

    $command = $Matches.command.Trim()
    if ($command -match '^[*?\s]+$') {
        return $false
    }

    $dangerousCommand = '(?i)^(rm|rmdir|del|erase|remove-item)\b|^git\s+(commit|push|reset|clean|checkout|switch)\b'
    return $command -notmatch $dangerousCommand
}

function Resolve-WorkerCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    if (Test-Path -LiteralPath $Command -PathType Leaf) {
        $resolvedFile = (Resolve-Path -LiteralPath $Command).Path
        if ([System.IO.Path]::GetExtension($resolvedFile) -ieq ".ps1") {
            return [pscustomobject]@{
                FileName        = (Get-Process -Id $PID).Path
                PrefixArguments = @("-NoProfile", "-File", $resolvedFile, "--")
            }
        }
        return [pscustomobject]@{
            FileName        = $resolvedFile
            PrefixArguments = @()
        }
    }

    $resolvedCommand = Get-Command $Command -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $resolvedCommand) {
        return $null
    }

    return [pscustomobject]@{
        FileName        = $resolvedCommand.Source
        PrefixArguments = @()
    }
}

function Invoke-ClaudeWorker {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskFile,
        [string]$WorkingDirectory = (Get-Location).Path,
        [ValidateSet("Analyze", "Implement")]
        [string]$Mode = "Analyze",
        [string[]]$AllowedBashTools = @(),
        [ValidateRange(1, 200)]
        [int]$MaxTurns = 30,
        [ValidateRange(1, 86400)]
        [int]$TimeoutSeconds = 1800,
        [string]$ClaudeCommand = "claude"
    )

    $skillRoot = Split-Path -Parent $PSScriptRoot
    $runsRoot = Join-Path $skillRoot "runs"

    try {
        if (-not (Test-Path -LiteralPath $TaskFile -PathType Leaf)) {
            return New-RejectedOutcome -Message "Task file does not exist: $TaskFile"
        }
        if (-not (Test-Path -LiteralPath $WorkingDirectory -PathType Container)) {
            return New-RejectedOutcome -Message "Working directory does not exist: $WorkingDirectory"
        }

        $resolvedTaskFile = (Resolve-Path -LiteralPath $TaskFile).Path
        $resolvedWorkingDirectory = (Resolve-Path -LiteralPath $WorkingDirectory).Path
        $taskContent = Get-Content -Raw -LiteralPath $resolvedTaskFile
        if ([string]::IsNullOrWhiteSpace($taskContent)) {
            return New-RejectedOutcome -Message "Task file is empty."
        }

        foreach ($tool in $AllowedBashTools) {
            if (-not (Test-ScopedBashTool -Tool $tool)) {
                return New-RejectedOutcome -Message "Each Bash allow rule must be scoped to a non-destructive command; unrestricted rules such as Bash(*) are not allowed."
            }
        }

        $workerCommand = Resolve-WorkerCommand -Command $ClaudeCommand
        if ($null -eq $workerCommand) {
            return New-RejectedOutcome -Message "Claude command was not found: $ClaudeCommand"
        }

        New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null
        $runId = "{0}-{1}" -f (Get-Date -Format "yyyyMMdd-HHmmss"), ([guid]::NewGuid().ToString("N").Substring(0, 8))
        $runDirectory = Join-Path $runsRoot $runId
        New-Item -ItemType Directory -Path $runDirectory | Out-Null

        $taskSnapshot = Join-Path $runDirectory "task.md"
        $requestFile = Join-Path $runDirectory "request.json"
        $stdoutFile = Join-Path $runDirectory "stdout.json"
        $stderrFile = Join-Path $runDirectory "stderr.txt"
        Set-Content -LiteralPath $taskSnapshot -Value $taskContent -Encoding utf8 -NoNewline

        $modeName = $Mode.ToLowerInvariant()
        $allowedTools = @("Read", "Glob", "Grep")
        $disallowedTools = @("WebFetch", "WebSearch")
        $permissionMode = "plan"

        if ($Mode -eq "Analyze") {
            $disallowedTools += @("Edit", "Write", "Bash")
        } else {
            $permissionMode = "acceptEdits"
            $allowedTools += @("Edit", "Write")
            $allowedTools += $AllowedBashTools
            $disallowedTools += @(
                "Bash(git commit *)",
                "Bash(git push *)",
                "Bash(git reset *)",
                "Bash(git clean *)",
                "Bash(git checkout *)",
                "Bash(git switch *)",
                "Bash(rm *)",
                "Bash(rmdir *)"
            )
        }

        $claudeArguments = @(
            "--print",
            "--output-format", "json",
            "--no-session-persistence",
            "--max-turns", $MaxTurns.ToString(),
            "--permission-mode", $permissionMode,
            "--allowedTools"
        ) + $allowedTools + @("--disallowedTools") + $disallowedTools

        $startedAt = [DateTimeOffset]::UtcNow
        $request = [ordered]@{
            run_id             = $runId
            mode               = $modeName
            working_directory  = $resolvedWorkingDirectory
            task_file          = $resolvedTaskFile
            task_snapshot      = $taskSnapshot
            claude_command     = $workerCommand.FileName
            max_turns          = $MaxTurns
            timeout_seconds    = $TimeoutSeconds
            allowed_bash_tools = @($AllowedBashTools)
            started_at         = $startedAt.ToString("o")
        }
        Set-Content -LiteralPath $requestFile -Value ($request | ConvertTo-Json -Depth 8) -Encoding utf8

        $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
        $startInfo.FileName = $workerCommand.FileName
        $startInfo.WorkingDirectory = $resolvedWorkingDirectory
        $startInfo.UseShellExecute = $false
        $startInfo.CreateNoWindow = $true
        $startInfo.RedirectStandardInput = $true
        $startInfo.RedirectStandardOutput = $true
        $startInfo.RedirectStandardError = $true

        foreach ($argument in @($workerCommand.PrefixArguments) + $claudeArguments) {
            $startInfo.ArgumentList.Add([string]$argument)
        }

        $process = [System.Diagnostics.Process]::new()
        $process.StartInfo = $startInfo
        if (-not $process.Start()) {
            throw "Failed to start Claude worker."
        }

        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $process.StandardInput.Write($taskContent)
        $process.StandardInput.Close()

        $completed = $process.WaitForExit($TimeoutSeconds * 1000)
        if (-not $completed) {
            $process.Kill($true)
            $process.WaitForExit()
        }

        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        $externalExitCode = if ($completed) { $process.ExitCode } else { $null }
        $process.Dispose()

        Set-Content -LiteralPath $stdoutFile -Value $stdout -Encoding utf8 -NoNewline
        Set-Content -LiteralPath $stderrFile -Value $stderr -Encoding utf8 -NoNewline

        $finishedAt = [DateTimeOffset]::UtcNow
        if (-not $completed) {
            return New-WorkerOutcome -ExitCode 124 -Payload ([ordered]@{
                status             = "timed_out"
                mode               = $modeName
                run_directory      = $runDirectory
                external_exit_code = $null
                timed_out          = $true
                started_at         = $startedAt.ToString("o")
                finished_at        = $finishedAt.ToString("o")
            })
        }

        $claudeResult = $null
        if (-not [string]::IsNullOrWhiteSpace($stdout)) {
            try {
                $claudeResult = $stdout | ConvertFrom-Json
            } catch {
                $claudeResult = $null
            }
        }

        $status = if ($externalExitCode -eq 0) { "completed" } else { "failed" }
        $payload = [ordered]@{
            status             = $status
            mode               = $modeName
            run_directory      = $runDirectory
            external_exit_code = $externalExitCode
            timed_out          = $false
            claude             = $claudeResult
            stdout_file        = $stdoutFile
            stderr_file        = $stderrFile
            started_at         = $startedAt.ToString("o")
            finished_at        = $finishedAt.ToString("o")
        }
        return New-WorkerOutcome -Payload $payload -ExitCode $externalExitCode
    } catch {
        return New-WorkerOutcome -ExitCode 1 -Payload ([ordered]@{
            status  = "failed"
            message = $_.Exception.Message
        })
    }
}

Export-ModuleMember -Function Invoke-ClaudeWorker
