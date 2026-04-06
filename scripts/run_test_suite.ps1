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
        if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
            throw "Expected .venv\Scripts\python.exe but it was not found. Run .\scripts\bootstrap_dev_env.ps1 first."
        }
        $ArgsList += "--require-venv"
        & .\.venv\Scripts\python.exe @ArgsList
    }
    else {
        py -3.12 @ArgsList
    }
}
finally {
    Pop-Location
}
