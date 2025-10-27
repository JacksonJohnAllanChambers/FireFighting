# PowerShell build helper for FireDrones
# Tries MSVC (cl), then MinGW g++, then LLVM clang++
# Usage:
#   ./build.ps1            # build firedrones.exe
#   ./build.ps1 -Clean     # remove outputs

param(
    [switch]$Clean
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$exe = "firedrones.exe"
$src = @(
    "src/Grid.cpp",
    "src/FireGen.cpp",
    "src/Drone.cpp",
    "src/Rescue.cpp",
    "src/Predict.cpp",
    "src/Simulation.cpp",
    "src/main.cpp"
)

if ($Clean) {
    if (Test-Path $exe) { Remove-Item $exe -Force }
    Get-ChildItem -Filter "*.obj" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned"
    exit 0
}

function Have($cmd) { Get-Command $cmd -ErrorAction SilentlyContinue | ForEach-Object { $_ } }

function Try-MSVC {
    if (-not (Have 'cl')) { return $false }
    Write-Host "Building with MSVC (cl)..."
    $args = @('/nologo','/std:c++20','/EHsc','/O2','/MD',"/Fe:$exe") + $src
    & cl @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host "MSVC build failed with exit code $LASTEXITCODE" -ForegroundColor Red
        return $false
    }
    return (Test-Path $exe)
}

function Try-GCC {
    if (-not (Have 'g++')) { return $false }
    Write-Host "Building with MinGW g++..."
    $args = @('-std=c++20','-O2','-o',$exe) + $src
    & g++ @args
    if ($LASTEXITCODE -eq 0 -and (Test-Path $exe)) { return $true }

    # Retry with stdc++fs for older GCC libstdc++
    Write-Host "Retrying g++ link with -lstdc++fs (older GCC)..."
    $args = @('-std=c++20','-O2','-o',$exe) + $src + '-lstdc++fs'
    & g++ @args
    return ($LASTEXITCODE -eq 0 -and (Test-Path $exe))
}

function Try-CLANG {
    if (-not (Have 'clang++')) { return $false }
    Write-Host "Building with LLVM clang++..."
    $args = @('-std=c++20','-O2','-o',$exe) + $src
    & clang++ @args
    return ($LASTEXITCODE -eq 0 -and (Test-Path $exe))
}

if (Try-MSVC) { Write-Host "Built $exe with MSVC" -ForegroundColor Green; exit 0 }
if (Try-GCC)  { Write-Host "Built $exe with g++"  -ForegroundColor Green; exit 0 }
if (Try-CLANG){ Write-Host "Built $exe with clang++" -ForegroundColor Green; exit 0 }

Write-Host "No suitable C++ compiler found. Options:" -ForegroundColor Yellow
Write-Host "  - Open 'x64 Native Tools Command Prompt for VS' (or Developer PowerShell) and rerun this script for MSVC."
Write-Host "  - Or install MinGW-w64 and ensure g++ is on PATH."
Write-Host "  - Or install LLVM and ensure clang++ is on PATH."
exit 1
