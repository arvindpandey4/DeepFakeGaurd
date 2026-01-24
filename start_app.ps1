# Start the Deepfake Detection App (Backend + Frontend)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Write-Host "🚀 Starting Deepfake Detection System..." -ForegroundColor Cyan

# 1. Start Backend (API)
Write-Host "Starting Backend (Port 8000)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "& '.\venv\Scripts\Activate.ps1'; python api.py" -WindowStyle Minimized

# Wait a moment for backend to initialize
Start-Sleep -Seconds 3

# 2. Start Frontend (UI)
Write-Host "Starting Frontend (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd frontend; npm run dev" -WindowStyle Minimized

# Wait for frontend to spin up
Start-Sleep -Seconds 4

# 3. Open Browser
Write-Host "Opening Dashboard..." -ForegroundColor Green
Start-Process "http://localhost:5173"

Write-Host "✅ System is running!" -ForegroundColor Green
Write-Host "   - Backend: http://localhost:8000"
Write-Host "   - Frontend: http://localhost:5173"
Write-Host "   (To stop everything, run .\stop_app.ps1)"
