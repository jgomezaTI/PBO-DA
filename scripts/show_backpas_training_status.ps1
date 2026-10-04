param(
    [Parameter(Mandatory = $true)]
    [string]$WorkDir,

    [ValidateRange(1, 100000)]
    [int]$Epochs = 200
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workPath = if ([System.IO.Path]::IsPathRooted($WorkDir)) {
    [System.IO.Path]::GetFullPath($WorkDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $WorkDir))
}
$checkpointDirectory = Join-Path $workPath "ml_training\graph_with_literals_8_GTR"
$trainingLog = Join-Path $checkpointDirectory "training_log.csv"

if (-not (Test-Path -LiteralPath $trainingLog -PathType Leaf)) {
    throw "Training log does not exist: $trainingLog"
}

$rows = @(Import-Csv -LiteralPath $trainingLog)
$validationRows = @($rows | Where-Object { $_.partition -eq "valid" })
$completedEpochs = $validationRows.Count
$percent = [math]::Round(100 * $completedEpochs / $Epochs, 1)

Write-Host "Completed epochs: $completedEpochs/$Epochs ($percent%)"
if ($validationRows.Count -gt 0) {
    $best = $validationRows | Sort-Object { [double]$_.loss } | Select-Object -First 1
    Write-Host "Best validation epoch: $($best.epoch)"
    Write-Host "Best validation loss: $($best.loss)"
}

$rows |
    Select-Object -Last 6 epoch, partition, time, loss, accuracy_micro, f1_score_macro |
    Format-Table -AutoSize

foreach ($checkpoint in @("best_model.pth", "last_model.pth", "last_optimizer.pth")) {
    $candidate = Join-Path $checkpointDirectory $checkpoint
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $item = Get-Item -LiteralPath $candidate
        Write-Host "${checkpoint}: present ($($item.LastWriteTime))"
    } else {
        Write-Host "${checkpoint}: missing"
    }
}
