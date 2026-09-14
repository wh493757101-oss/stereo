Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$receivedArguments = @($args)
$task = [Console]::In.ReadToEnd()

if ($task -like "*SIMULATE_TIMEOUT*") {
    Start-Sleep -Seconds 3
}

if ($task -like "*SIMULATE_FAILURE*") {
    [Console]::Error.WriteLine("fixture failure")
    exit 7
}

if ($task -like "*SIMULATE_INVALID_JSON*") {
    [Console]::Out.WriteLine("not-json")
    exit 0
}

$response = [ordered]@{
    type      = "result"
    subtype   = "success"
    is_error  = $false
    result    = "fixture completed"
    arguments = $receivedArguments
}

[Console]::Out.WriteLine(($response | ConvertTo-Json -Depth 8 -Compress))
exit 0
