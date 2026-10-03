Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
	Write-Error 'Conda is not installed or not initialized for PowerShell. Please run install\install-conda.ps1 first.'
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $ProjectRoot

try {
	try { conda deactivate | Out-Null } catch { }

	conda remove -n ai_model_server --all -y | Out-Null

	Write-Output 'Creating conda environment (environment-windows.yml)...'
	conda env create -f (Join-Path $PSScriptRoot 'environment-windows.yml')

	Write-Output "Activating environment ai_model_server..."
	conda activate ai_model_server

	Write-Output 'Running migration'
	python (Join-Path $ProjectRoot 'migrate.py')

	Write-Output 'Starting server...'
	python (Join-Path $ProjectRoot 'server.py')
}
finally {
	Pop-Location
}