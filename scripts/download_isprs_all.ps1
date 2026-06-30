$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$defaultRoot = Resolve-Path (Join-Path $scriptDir "..")
$pythonDl = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$root = if ($env:PROJECT_DIR) { $env:PROJECT_DIR } else { $defaultRoot.Path }
$password = $env:ISPRS_SHARE_PASSWORD

if (-not $password) {
  throw "Set ISPRS_SHARE_PASSWORD before downloading password-protected ISPRS archives."
}

Set-Location $root

function Test-ZipFile {
  param(
    [Parameter(Mandatory = $true)][string]$Path
  )
  $out = & $pythonDl -c "import sys, zipfile; p=sys.argv[1]; z=zipfile.ZipFile(p,'r'); z.testzip(); print('OK')" $Path 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "Bad zip: $Path`n$out"
  }
}

function Download-And-Prepare {
  param(
    [Parameter(Mandatory = $true)][string]$ShareUrl,
    [Parameter(Mandatory = $true)][string]$ZipPath,
    [Parameter(Mandatory = $true)][string]$DatasetName,
    [Parameter(Mandatory = $true)][string]$PreparedDir
  )

  New-Item -ItemType Directory -Force -Path (Split-Path $ZipPath) | Out-Null
  New-Item -ItemType Directory -Force -Path $PreparedDir | Out-Null

  $maxAttempts = 999
  for ($i = 1; $i -le $maxAttempts; $i++) {
    Write-Host "[${DatasetName}] download attempt $i"
    & $pythonDl "scripts/download_seafile_shared.py" `
      --share-url $ShareUrl `
      --password $password `
      --output $ZipPath `
      --resume `
      --retries 30 `
      --retry-delay 5

    Write-Host "[${DatasetName}] validating zip file..."
    try {
      Test-ZipFile -Path $ZipPath
      Write-Host "[${DatasetName}] zip OK"
      break
    } catch {
      Write-Host "[${DatasetName}] zip invalid, retrying..."
      Start-Sleep -Seconds 10
    }
  }

  Write-Host "[${DatasetName}] preparing dataset..."
  & $pythonDl "scripts/prepare_isprs.py" `
    --dataset $DatasetName `
    --raw-zip $ZipPath `
    --output-root $PreparedDir `
    --overwrite
}

Download-And-Prepare `
  -ShareUrl "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/" `
  -ZipPath "data_raw/Potsdam.zip" `
  -DatasetName "potsdam" `
  -PreparedDir "data/Potsdam_prepared"

Download-And-Prepare `
  -ShareUrl "https://seafile.projekt.uni-hannover.de/f/6a06a837b1f349cfa749/" `
  -ZipPath "data_raw/Vaihingen.zip" `
  -DatasetName "vaihingen" `
  -PreparedDir "data/Vaihingen_prepared"
