param(
    [string]$OutputDirectory = "thesis\pdfs"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$mainFile = Join-Path $repositoryRoot "thesis\main.tex"
$miktexBin = Join-Path $repositoryRoot ".tmp\miktex\portable\texmfs\install\miktex\bin\x64"
$gitPerlBin = "C:\Program Files\Git\usr\bin"
$latexmk = Join-Path $miktexBin "latexmk.exe"

if (-not (Test-Path $mainFile)) {
    throw "Missing thesis source: $mainFile"
}
if (-not (Test-Path $latexmk)) {
    throw "MiKTeX Portable is not installed at $miktexBin"
}

$outputPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $OutputDirectory))
$tempPath = Join-Path $repositoryRoot ".tmp\latex-temp"
New-Item -ItemType Directory -Force $outputPath | Out-Null
New-Item -ItemType Directory -Force $tempPath | Out-Null

$env:PATH = "$gitPerlBin;$miktexBin;$env:PATH"
$env:TEMP = $tempPath
$env:TMP = $tempPath

Push-Location (Split-Path -Parent $mainFile)
try {
    & $latexmk `
        "-norc" `
        "-pdf" `
        "-interaction=nonstopmode" `
        "-halt-on-error" `
        "-synctex=1" `
        "-outdir=$outputPath" `
        $mainFile
    $latexmkExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

if ($latexmkExitCode -ne 0) {
    exit $latexmkExitCode
}

Write-Host "PDF generated at: $(Join-Path $outputPath 'main.pdf')"
