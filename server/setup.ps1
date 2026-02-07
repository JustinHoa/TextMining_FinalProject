# Stop on error
$ErrorActionPreference = "Stop"

Write-Host "============================"
Write-Host "   ViFactCheck - AUTO SETUP  "
Write-Host "============================"

# 1. Environment Check
if (-not (Test-Path ".env")) {
    Write-Host "!! Warning: .env file not found. Creating from .env.example..." -ForegroundColor Yellow
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Host "== Created .env. Please update your API Keys!" -ForegroundColor Green
    } else {
        New-Item ".env" -ItemType File
        Set-Content .env "OPENAI_API_KEY=`nSERPER_API_KEY=`nHF_TOKEN="
        Write-Host "== Created empty .env. Please fill in your keys." -ForegroundColor Green
    }
} else {
    Write-Host "== .env file exists."
}

# 2. Virtual Environment
if (-not (Test-Path "venv")) {
    Write-Host "== Creating virtual environment..."
    python -m venv venv
    Write-Host "== venv created."
}

# Activate venv (PowerShell specific)
$venvScript = ".\venv\Scripts\Activate.ps1"
if (Test-Path $venvScript) {
    Write-Host "== Activating virtual environment..."
    & $venvScript
} else {
    Write-Host "!! Could not find activate script. Assuming you are already in venv or global python." -ForegroundColor Yellow
}

# 3. Dependencies
Write-Host "== Installing dependencies..."
pip install -r requirements.txt
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Optional: Install PyTorch with CUDA if needed
# Re-install qdrant-client to be safe
pip install --upgrade qdrant-client openai fastapi uvicorn python-dotenv sentence-transformers trafilatura

# 4. Data Seeding
Write-Host "== Running seeding script (Download Dataset)..."
if (Test-Path "scripts/seeding.py") {
    python scripts/seeding.py
} else {
    Write-Host "!! Error: scripts/seeding.py not found!" -ForegroundColor Red
    exit 1
}

# 5. Embedding
Write-Host "== Running embedding script (This may take a while)..."
if (Test-Path "scripts/embedding.py") {
    python scripts/embedding.py
} else {
     Write-Host "!! Error: scripts/embedding.py not found!" -ForegroundColor Red
     exit 1
}

Write-Host "============================"
Write-Host "== SETUP COMPLETED SUCCESSFULLY!"
Write-Host "Run the server with: uvicorn main:app --reload"
Write-Host "============================"