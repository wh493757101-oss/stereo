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

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Import-Module (Join-Path $PSScriptRoot "ClaudeWorker.psm1") -Force
$outcome = Invoke-ClaudeWorker @PSBoundParameters
[Console]::Out.WriteLine(($outcome.Payload | ConvertTo-Json -Depth 20 -Compress))
exit $outcome.ExitCode
