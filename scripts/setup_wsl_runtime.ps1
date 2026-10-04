$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$portableRepositoryRoot = $repositoryRoot.Replace("\", "/")
$wslRepositoryRoot = (wsl --distribution Ubuntu -- wslpath -a $portableRepositoryRoot).Trim()
if (-not $wslRepositoryRoot) {
    throw "Unable to translate the repository path for WSL."
}

wsl --distribution Ubuntu --user root -- apt-get update
if ($LASTEXITCODE -ne 0) {
    throw "Unable to update Ubuntu packages."
}
wsl --distribution Ubuntu --user root -- apt-get install --yes build-essential ca-certificates curl bzip2 wget
if ($LASTEXITCODE -ne 0) {
    throw "Unable to install Ubuntu build tools."
}

wsl --distribution Ubuntu -- bash "$wslRepositoryRoot/scripts/setup_wsl_runtime.sh" $wslRepositoryRoot
if ($LASTEXITCODE -ne 0) {
    throw "WSL runtime setup failed with exit code $LASTEXITCODE."
}
