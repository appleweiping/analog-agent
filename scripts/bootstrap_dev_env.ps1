param(
    [string]$PythonVersion = "3.12"
)

$ErrorActionPreference = "Stop"

function Clear-TransientBootstrapEnv {
    param(
        [string[]]$Names
    )

    $removed = @{}
    foreach ($name in $Names) {
        $value = [Environment]::GetEnvironmentVariable($name, "Process")
        if ($null -ne $value -and $value -ne "") {
            $removed[$name] = $value
            [Environment]::SetEnvironmentVariable($name, $null, "Process")
        }
    }

    return $removed
}

function Restore-TransientBootstrapEnv {
    param(
        [hashtable]$Values
    )

    foreach ($entry in $Values.GetEnumerator()) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
    }
}

function Resolve-ExtraPipArgs {
    $args = @()
    $indexUrl = [Environment]::GetEnvironmentVariable("PIP_INDEX_URL", "Process")
    if ([string]::IsNullOrWhiteSpace($indexUrl)) {
        $args += @("-i", "https://pypi.org/simple")
    }

    return $args
}

Write-Host "Using interpreter: py -$PythonVersion"
$envVarsToClear = @(
    "PIP_NO_INDEX",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY"
)
$clearedEnv = Clear-TransientBootstrapEnv -Names $envVarsToClear
$pipArgs = Resolve-ExtraPipArgs

if ($clearedEnv.Count -gt 0) {
    Write-Host "Temporarily cleared environment overrides that can block pip:" ($clearedEnv.Keys -join ", ")
}

try {
    py -$PythonVersion -c "import sys; print(sys.executable)"
    py -$PythonVersion -m pip --version
    py -$PythonVersion -m venv .venv
    & .\.venv\Scripts\python.exe -c "import sys; print(sys.executable)"
    & .\.venv\Scripts\python.exe -m pip --version
    & .\.venv\Scripts\python.exe -m pip install @pipArgs -U pip setuptools wheel
    & .\.venv\Scripts\python.exe -m pip install @pipArgs -e ".[dev]"
    & .\.venv\Scripts\python.exe -c "import fastapi, httpx, pytest; print('API dependencies ready')"
}
finally {
    Restore-TransientBootstrapEnv -Values $clearedEnv
}

Write-Host "Development environment is ready in .venv"
