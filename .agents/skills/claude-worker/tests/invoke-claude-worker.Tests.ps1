Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$testRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$skillRoot = Split-Path -Parent $testRoot
$workerScript = Join-Path $skillRoot "scripts\invoke-claude-worker.ps1"
$workerModule = Join-Path $skillRoot "scripts\ClaudeWorker.psm1"
$fakeClaude = Join-Path $testRoot "fake-claude.ps1"
$testWorkspace = Join-Path $testRoot "workspace"
$taskFile = Join-Path $testWorkspace "task.md"

Import-Module $workerModule -Force

function Reset-TestWorkspace {
    if (Test-Path -LiteralPath $testWorkspace) {
        Remove-Item -LiteralPath $testWorkspace -Recurse -Force
    }
    New-Item -ItemType Directory -Path $testWorkspace | Out-Null
}

function Invoke-WrapperProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$WorkerArguments
    )

    $pwsh = (Get-Process -Id $PID).Path
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $pwsh
    $startInfo.WorkingDirectory = $testWorkspace
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    foreach ($argument in @("-NoProfile", "-File", $workerScript) + $WorkerArguments) {
        $startInfo.ArgumentList.Add([string]$argument)
    }

    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    $process.Start() | Out-Null
    $stdout = $process.StandardOutput.ReadToEndAsync()
    $stderr = $process.StandardError.ReadToEndAsync()
    $process.WaitForExit()

    [pscustomobject]@{
        ExitCode = $process.ExitCode
        Stdout   = $stdout.GetAwaiter().GetResult()
        Stderr   = $stderr.GetAwaiter().GetResult()
    }
}

Describe "Invoke-ClaudeWorker" {
    BeforeEach {
        Reset-TestWorkspace
    }

    It "runs analysis with read-only tools and stores artifacts inside the skill" {
        Set-Content -LiteralPath $taskFile -Value "Inspect the project and report findings." -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Analyze"
            TimeoutSeconds = 10
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 0
        $result = $execution.Payload
        $result.status | Should Be "completed"
        $result.mode | Should Be "analyze"
        $result.claude.result | Should Be "fixture completed"
        $result.run_directory.StartsWith((Join-Path $skillRoot "runs")) | Should Be $true
        (Test-Path -LiteralPath (Join-Path $result.run_directory "task.md")) | Should Be $true
        (Test-Path -LiteralPath (Join-Path $result.run_directory "stdout.json")) | Should Be $true
        (Test-Path -LiteralPath (Join-Path $result.run_directory "stderr.txt")) | Should Be $true
        ($result.claude.arguments -contains "plan") | Should Be $true
        ($result.claude.arguments -contains "Read") | Should Be $true
        ($result.claude.arguments -contains "Glob") | Should Be $true
        ($result.claude.arguments -contains "Grep") | Should Be $true
        ($result.claude.arguments -contains "Edit") | Should Be $true
        ($result.claude.arguments -contains "Write") | Should Be $true
        ($result.claude.arguments -contains "Bash") | Should Be $true
        ($result.claude.arguments -contains "WebFetch") | Should Be $true
        ($result.claude.arguments -contains "WebSearch") | Should Be $true
    }

    It "allows edits and only explicitly scoped Bash tools in implement mode" {
        Set-Content -LiteralPath $taskFile -Value "Implement the bounded change." -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Implement"
            AllowedBashTools = "Bash(python -m pytest *)"
            TimeoutSeconds = 10
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 0
        $result = $execution.Payload
        $result.status | Should Be "completed"
        $result.mode | Should Be "implement"
        ($result.claude.arguments -contains "acceptEdits") | Should Be $true
        ($result.claude.arguments -contains "Edit") | Should Be $true
        ($result.claude.arguments -contains "Write") | Should Be $true
        ($result.claude.arguments -contains "Bash(python -m pytest *)") | Should Be $true
        ($result.claude.arguments -contains "Bash(git commit *)") | Should Be $true
        ($result.claude.arguments -contains "Bash(git push *)") | Should Be $true
        ($result.claude.arguments -contains "WebFetch") | Should Be $true
        ($result.claude.arguments -contains "WebSearch") | Should Be $true
        ($result.claude.arguments -contains "Bash(*)") | Should Be $false
    }

    It "rejects an unrestricted Bash allow rule" {
        Set-Content -LiteralPath $taskFile -Value "Implement a change." -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Implement"
            AllowedBashTools = "Bash(*)"
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 2
        $execution.Payload.status | Should Be "rejected"
        $execution.Payload.message | Should Match "scoped"
    }

    It "propagates an external worker failure and preserves stderr" {
        Set-Content -LiteralPath $taskFile -Value "SIMULATE_FAILURE" -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Analyze"
            TimeoutSeconds = 10
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 7
        $execution.Payload.status | Should Be "failed"
        $execution.Payload.external_exit_code | Should Be 7
        (Get-Content -Raw -LiteralPath (Join-Path $execution.Payload.run_directory "stderr.txt")) | Should Match "fixture failure"
    }

    It "terminates a worker that exceeds the timeout" {
        Set-Content -LiteralPath $taskFile -Value "SIMULATE_TIMEOUT" -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Analyze"
            TimeoutSeconds = 1
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 124
        $execution.Payload.status | Should Be "timed_out"
        $execution.Payload.timed_out | Should Be $true
    }

    It "rejects an empty task without starting the worker" {
        Set-Content -LiteralPath $taskFile -Value "   " -Encoding utf8
        $execution = Invoke-ClaudeWorker -TaskFile $taskFile -WorkingDirectory $testWorkspace -ClaudeCommand $fakeClaude

        $execution.ExitCode | Should Be 2
        $execution.Payload.status | Should Be "rejected"
        $execution.Payload.message | Should Match "empty"
    }

    It "rejects a missing task file" {
        $execution = Invoke-ClaudeWorker -TaskFile (Join-Path $testWorkspace "missing.md") -WorkingDirectory $testWorkspace -ClaudeCommand $fakeClaude

        $execution.ExitCode | Should Be 2
        $execution.Payload.message | Should Match "does not exist"
    }

    It "rejects a missing working directory" {
        Set-Content -LiteralPath $taskFile -Value "Analyze." -Encoding utf8
        $execution = Invoke-ClaudeWorker -TaskFile $taskFile -WorkingDirectory (Join-Path $testWorkspace "missing") -ClaudeCommand $fakeClaude

        $execution.ExitCode | Should Be 2
        $execution.Payload.message | Should Match "Working directory"
    }

    It "rejects a missing Claude command" {
        Set-Content -LiteralPath $taskFile -Value "Analyze." -Encoding utf8
        $execution = Invoke-ClaudeWorker -TaskFile $taskFile -WorkingDirectory $testWorkspace -ClaudeCommand "missing-claude-command-for-test"

        $execution.ExitCode | Should Be 2
        $execution.Payload.message | Should Match "not found"
    }

    It "rejects a destructive Bash allow rule" {
        Set-Content -LiteralPath $taskFile -Value "Implement." -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Implement"
            AllowedBashTools = "Bash(git push origin main)"
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 2
        $execution.Payload.status | Should Be "rejected"
    }

    It "keeps a successful non-JSON worker response as an artifact" {
        Set-Content -LiteralPath $taskFile -Value "SIMULATE_INVALID_JSON" -Encoding utf8
        $parameters = @{
            TaskFile = $taskFile
            WorkingDirectory = $testWorkspace
            ClaudeCommand = $fakeClaude
            Mode = "Analyze"
            TimeoutSeconds = 10
        }
        $execution = Invoke-ClaudeWorker @parameters

        $execution.ExitCode | Should Be 0
        $execution.Payload.status | Should Be "completed"
        $execution.Payload.claude | Should Be $null
        (Get-Content -Raw -LiteralPath $execution.Payload.stdout_file) | Should Match "not-json"
    }

    It "exposes the module result through the command-line wrapper" {
        Set-Content -LiteralPath $taskFile -Value "Wrapper smoke test." -Encoding utf8
        $execution = Invoke-WrapperProcess -WorkerArguments @(
            "-TaskFile", $taskFile,
            "-WorkingDirectory", $testWorkspace,
            "-ClaudeCommand", $fakeClaude,
            "-Mode", "Analyze",
            "-TimeoutSeconds", "10"
        )

        $execution.ExitCode | Should Be 0
        $result = $execution.Stdout | ConvertFrom-Json
        $result.status | Should Be "completed"
        $result.claude.result | Should Be "fixture completed"
    }
}
