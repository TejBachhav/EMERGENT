#!/usr/bin/env pwsh
# CyberGuard AI - Project Startup Script
# This script starts both the backend and frontend services

Write-Host "🛡️  Starting CyberGuard AI Project" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

# Function to check if a process is running on a specific port
function Test-Port {
    param($Port)
    $connection = Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue
    return $connection
}

# Check if required tools are installed
Write-Host "🔍 Checking prerequisites..." -ForegroundColor Yellow

# Check Node.js
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js from https://nodejs.org/" -ForegroundColor Red
    exit 1
}

# Check Python
try {
    $pythonVersion = python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python from https://python.org/" -ForegroundColor Red
    exit 1
}

# Check MongoDB
if (Test-Port -Port 27017) {
    Write-Host "✅ MongoDB: Running on port 27017" -ForegroundColor Green
} else {
    Write-Host "⚠️  MongoDB not detected on port 27017. Please ensure MongoDB is running." -ForegroundColor Yellow
}

# Check Ollama
try {
    $ollamaStatus = ollama --version
    Write-Host "✅ Ollama: $ollamaStatus" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Ollama not found. Please install Ollama from https://ollama.ai/" -ForegroundColor Yellow
}

Write-Host "`n🔧 Installing dependencies..." -ForegroundColor Yellow

# Install backend dependencies
Write-Host "📦 Installing backend dependencies..." -ForegroundColor Blue
Set-Location backend
if (Test-Path "requirements_clean.txt") {
    pip install -r requirements_clean.txt
} else {
    pip install -r requirements.txt
}

# Install frontend dependencies
Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Blue
Set-Location ../frontend
npm install

Set-Location ..

Write-Host "`n🚀 Starting services..." -ForegroundColor Yellow

# Start backend in a new PowerShell window
Write-Host "🔙 Starting backend server on port 9000..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\backend'; python app.py"

# Wait a moment for backend to start
Start-Sleep -Seconds 3

# Start frontend in a new PowerShell window
Write-Host "🌐 Starting frontend server on port 5000..." -ForegroundColor Blue
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\frontend'; npm start"

Write-Host "`n✅ CyberGuard AI started successfully!" -ForegroundColor Green
Write-Host "🌐 Frontend: http://localhost:5000" -ForegroundColor Cyan
Write-Host "🔙 Backend: http://localhost:9000" -ForegroundColor Cyan
Write-Host "📊 Health Check: http://localhost:9000/api/health" -ForegroundColor Cyan

Write-Host "`n📝 Next steps:" -ForegroundColor Yellow
Write-Host "1. Open http://localhost:5000 in your browser" -ForegroundColor White
Write-Host "2. Register a new account or login" -ForegroundColor White
Write-Host "3. Start chatting with CyberGuard AI!" -ForegroundColor White

Write-Host "`n⚙️  Optional Configuration:" -ForegroundColor Yellow
Write-Host "- Set GEMINI_API_KEY environment variable for enhanced AI responses" -ForegroundColor White
Write-Host "- Ensure Ollama has the 'llama3' model installed: ollama pull llama3" -ForegroundColor White

Read-Host "`nPress Enter to exit this script (services will continue running)"
