# ============================================================================
# PaddleOCR-VL Environment Setup Script (Windows)
# ============================================================================
# This script sets up a dedicated virtual environment for PaddleOCR-VL
# with support for both CPU and GPU inference.
#
# Usage:
#   .\setup_paddleocr.ps1 -Mode cpu    # For CPU-only systems
#   .\setup_paddleocr.ps1 -Mode gpu    # For CUDA 12.6 GPU systems
#   .\setup_paddleocr.ps1              # Interactive mode (asks you)
# ============================================================================

param(
    [ValidateSet("cpu", "gpu", "")]
    [string]$Mode = ""
)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  PaddleOCR-VL Environment Setup Script    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Determine mode if not specified
if ($Mode -eq "") {
    Write-Host "Select installation mode:" -ForegroundColor Yellow
    Write-Host "  [1] CPU only (works on any system)"
    Write-Host "  [2] GPU with CUDA 12.6 (requires NVIDIA GPU)"
    Write-Host ""
    $choice = Read-Host "Enter choice (1 or 2)"
    
    if ($choice -eq "1") {
        $Mode = "cpu"
    } elseif ($choice -eq "2") {
        $Mode = "gpu"
    } else {
        Write-Host "Invalid choice. Defaulting to CPU mode." -ForegroundColor Red
        $Mode = "cpu"
    }
}

Write-Host ""
Write-Host "Selected mode: $Mode" -ForegroundColor Green
Write-Host ""

# Get the script's directory (project root)
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPath = Join-Path $ProjectRoot ".venv_paddleocr"

# Step 1: Create virtual environment
Write-Host "[1/5] Creating virtual environment..." -ForegroundColor Yellow
if (Test-Path $VenvPath) {
    Write-Host "  Virtual environment already exists at: $VenvPath" -ForegroundColor Gray
    $overwrite = Read-Host "  Overwrite? (y/n)"
    if ($overwrite -eq "y") {
        Remove-Item -Recurse -Force $VenvPath
        python -m venv $VenvPath
        Write-Host "  Created fresh virtual environment." -ForegroundColor Green
    }
} else {
    python -m venv $VenvPath
    Write-Host "  Created virtual environment at: $VenvPath" -ForegroundColor Green
}

# Step 2: Activate virtual environment
Write-Host ""
Write-Host "[2/5] Activating virtual environment..." -ForegroundColor Yellow
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"
& $ActivateScript
Write-Host "  Activated." -ForegroundColor Green

# Step 3: Upgrade pip
Write-Host ""
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& "$VenvPath\Scripts\python.exe" -m pip install --upgrade pip
Write-Host "  pip upgraded." -ForegroundColor Green

# Step 4: Install PaddlePaddle
Write-Host ""
Write-Host "[4/5] Installing PaddlePaddle ($Mode mode)..." -ForegroundColor Yellow

if ($Mode -eq "gpu") {
    Write-Host "  Installing PaddlePaddle GPU (CUDA 12.6)..." -ForegroundColor Gray
    & "$VenvPath\Scripts\pip.exe" install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
} else {
    Write-Host "  Installing PaddlePaddle CPU..." -ForegroundColor Gray
    & "$VenvPath\Scripts\pip.exe" install paddlepaddle==3.2.1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: PaddlePaddle installation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  PaddlePaddle installed." -ForegroundColor Green

# Step 5: Install PaddleOCR with doc-parser
Write-Host ""
Write-Host "[5/5] Installing PaddleOCR with doc-parser..." -ForegroundColor Yellow
& "$VenvPath\Scripts\pip.exe" install -U "paddleocr[doc-parser]"

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: PaddleOCR installation failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  PaddleOCR installed." -ForegroundColor Green

# Step 6: Install Windows-specific safetensors
Write-Host ""
Write-Host "[BONUS] Installing Windows-specific safetensors..." -ForegroundColor Yellow
& "$VenvPath\Scripts\pip.exe" install https://xly-devops.cdn.bcebos.com/safetensors-nightly/safetensors-0.6.2.dev0-cp38-abi3-win_amd64.whl

if ($LASTEXITCODE -ne 0) {
    Write-Host "  WARNING: safetensors installation had issues (may still work)." -ForegroundColor Yellow
} else {
    Write-Host "  safetensors installed." -ForegroundColor Green
}

# Verification
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Verifying Installation                    " -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

$VerifyScript = @"
import sys
print(f"Python: {sys.version}")

try:
    import paddle
    print(f"PaddlePaddle: {paddle.__version__}")
    print(f"  Device: {'GPU' if paddle.is_compiled_with_cuda() else 'CPU'}")
except ImportError as e:
    print(f"ERROR: PaddlePaddle import failed: {e}")
    sys.exit(1)

try:
    from paddleocr import PaddleOCRVL
    print("PaddleOCR-VL: OK (import successful)")
except ImportError as e:
    print(f"ERROR: PaddleOCR-VL import failed: {e}")
    sys.exit(1)

print("")
print("SUCCESS! PaddleOCR-VL is ready to use.")
"@

$VerifyScript | & "$VenvPath\Scripts\python.exe" -

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  INSTALLATION COMPLETE!                    " -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To activate this environment in the future:" -ForegroundColor Yellow
    Write-Host "  .\.venv_paddleocr\Scripts\Activate.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "To test PaddleOCR-VL:" -ForegroundColor Yellow
    Write-Host "  python test_paddleocr.py" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host "  INSTALLATION VERIFICATION FAILED          " -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please check the error messages above." -ForegroundColor Yellow
    exit 1
}
