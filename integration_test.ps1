Write-Host "🛡️ Testing CyberGuard AI Integration"
Write-Host "======================================"

Write-Host ""
Write-Host "1. Testing Frontend Availability..."
$response = Invoke-WebRequest -Uri http://localhost:5000 -UseBasicParsing -ErrorAction SilentlyContinue
if ($response.StatusCode -eq 200) {
    Write-Host "✅ Frontend running on localhost:5000"
} else {
    Write-Host "❌ Frontend not accessible: HTTP $($response.StatusCode)"
}

Write-Host ""
Write-Host "2. Testing Backend Health..."
$response = Invoke-WebRequest -Uri http://localhost:9000/api/health -UseBasicParsing -ErrorAction SilentlyContinue
if ($response.Content -match '"status":\s*"healthy"') {
    Write-Host "✅ Backend healthy on localhost:9000"
    Write-Host "Backend response: $($response.Content)"
} else {
    Write-Host "❌ Backend not healthy"
}

Write-Host ""
Write-Host "3. Testing Chat Endpoint..."
$response = Invoke-WebRequest -Uri http://localhost:9000/stream_chat -Method POST -Body '{"message": "Hello"}' -ContentType "application/json" -UseBasicParsing -ErrorAction SilentlyContinue
if ($response.Content) {
    Write-Host "✅ Chat endpoint responding"
    Write-Host "Response preview: $($response.Content.Substring(0, 100))..."
} else {
    Write-Host "❌ Chat endpoint not responding"
}

Write-Host ""
Write-Host "4. Environment Configuration..."
Write-Host "REACT_APP_BACKEND_URL in frontend/.env:"
Get-Content -Path "c:\DOCUMENTS\GITHUB\EMERGENT\frontend\.env" | Select-String "REACT_APP_BACKEND_URL"
