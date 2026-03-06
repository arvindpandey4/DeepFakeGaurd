# Stop the Deepfake Detection App

Write-Host "Stopping Deepfake Detection System..." -ForegroundColor Red

# Function to stop process by port
function Stop-PortProcess($port) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    
    if ($connections) {
        $pids = $connections.OwningProcess | Select-Object -Unique | Where-Object { $_ -gt 0 }
        
        foreach ($processId in $pids) {
            Write-Host "   - Killing process on port $port (PID: $processId)..." -ForegroundColor Yellow
            Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
        }
    }
    else {
        Write-Host "   - Port $port is already clear." -ForegroundColor Gray
    }
}

# 1. Stop Backend (Port 8000)
Write-Host "Checking Backend (Port 8000)..."
Stop-PortProcess 8000

# 2. Stop Frontend (Port 5173 - Vite)
Write-Host "Checking Frontend (Port 5173)..."
Stop-PortProcess 5173

# 3. Stop Node processes (Cleanup)
Write-Host "Cleaning up Node processes..."
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# 4. Stop Python processes (Cleanup)
Write-Host "Cleaning up Python processes..."
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*api.py*" } | Stop-Process -Force -ErrorAction SilentlyContinue

Write-Host "System stopped successfully." -ForegroundColor Green
