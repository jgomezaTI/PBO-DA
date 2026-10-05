[CmdletBinding()]
param(
    [string]$Dataset = "data\backpas\MVC",
    [string]$Results = "results\MVC",
    [string]$Output = "artifacts\mvc-training-payload.tar.gz"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$datasetPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Dataset))
$resultsPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Results))
$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Output))

foreach ($required in @(
    (Join-Path $datasetPath "partitions.json"),
    (Join-Path $datasetPath "ml_dataset_literals"),
    (Join-Path $resultsPath "baseline\seed-0\partitions.pkl"),
    (Join-Path $resultsPath "smoke\seed-0\partitions.pkl")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required training payload input is missing: $required"
    }
}

New-Item -ItemType Directory -Force (Split-Path -Parent $outputPath) | Out-Null
Push-Location $repositoryRoot
try {
    $datasetArchivePath = $Dataset.Replace("\", "/")
    $resultsArchivePath = $Results.Replace("\", "/")
    tar -czf $outputPath $datasetArchivePath $resultsArchivePath
    if ($LASTEXITCODE -ne 0) {
        throw "tar failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash.ToLowerInvariant()
$checksumPath = "$outputPath.sha256"
Set-Content -LiteralPath $checksumPath -Value "$hash  $([System.IO.Path]::GetFileName($outputPath))" -Encoding ascii
Write-Host "Payload: $outputPath"
Write-Host "SHA-256: $hash"
Write-Host "Checksum: $checksumPath"
