param(
    [Parameter(Mandatory = $true)]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [string]$WorkDir,

    [ValidateRange(1, 100000)]
    [int]$Epochs = 200,

    [int]$Seed = 0,

    [ValidateRange(1, 256)]
    [int]$Threads = 4,

    [string]$Distribution = "Ubuntu",

    [switch]$Resume
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$cli = Join-Path $repositoryRoot ".venv\Scripts\backbone-pbo.exe"
if (-not (Test-Path -LiteralPath $cli)) {
    throw "backbone-pbo is not installed at $cli. Run uv sync first."
}

function Resolve-RepositoryPath([string]$Value) {
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Value))
}

$datasetPath = Resolve-RepositoryPath $Dataset
$workPath = Resolve-RepositoryPath $WorkDir
if (-not (Test-Path -LiteralPath $datasetPath -PathType Container)) {
    throw "Dataset directory does not exist: $datasetPath"
}

if ($Resume) {
    $checkpointDirectory = Join-Path $workPath "ml_training\graph_with_literals_8_GTR"
    foreach ($required in @("training_log.csv", "last_model.pth", "last_optimizer.pth")) {
        $candidate = Join-Path $checkpointDirectory $required
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            throw "Cannot resume because the checkpoint file is missing: $candidate"
        }
    }
}

$arguments = @(
    "train-backpas",
    $datasetPath,
    "--work-dir", $workPath,
    "--epochs", $Epochs,
    "--seed", $Seed,
    "--threads", $Threads,
    "--distribution", $Distribution
)
if ($Resume) {
    $arguments += "--resume"
}

Write-Host ("{0} BackPaS training: dataset={1}, work_dir={2}, epochs={3}, seed={4}" -f `
    $(if ($Resume) { "Resuming" } else { "Starting" }),
    $datasetPath,
    $workPath,
    $Epochs,
    $Seed)

& $cli @arguments
exit $LASTEXITCODE
