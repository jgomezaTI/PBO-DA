[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$externalRoot = Join-Path $repositoryRoot "external"

New-Item -ItemType Directory -Force -Path $externalRoot | Out-Null

$repositories = @(
    @{
        Name = "backpas"
        Url = "https://github.com/bryan-alvarado-ulloa/backpas.git"
    },
    @{
        Name = "guroback"
        Url = "https://github.com/bryan-alvarado-ulloa/guroback.git"
    }
)

foreach ($repository in $repositories) {
    $destination = Join-Path $externalRoot $repository.Name
    if (Test-Path (Join-Path $destination ".git")) {
        Write-Host "Keeping existing checkout: $destination"
        git -C $destination remote get-url origin
        git -C $destination rev-parse HEAD
        continue
    }

    if (Test-Path $destination) {
        throw "Destination exists but is not a Git checkout: $destination"
    }

    Write-Host "Cloning $($repository.Url) into $destination"
    git clone $repository.Url $destination
}
