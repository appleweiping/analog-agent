param(
    [switch]$RequireApiDeps,
    [switch]$UseVenv
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$ArgsList = @("scripts\run_test_suite.py")

if ($RequireApiDeps) {
    $ArgsList += "--require-api-deps"
}

Push-Location $RepoRoot
try {
    if ($UseVenv) {
        & .\.venv\Scripts\python.exe @ArgsList
    }
    else {
        py -3.12 @ArgsList
    }
}
finally {
    Pop-Location
}
