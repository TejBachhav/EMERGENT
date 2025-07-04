#!/bin/sh
set -e

# Security: Check file permissions and integrity
if [ ! -f "/backend/app.py" ]; then
    echo "ERROR: Backend application not found!"
    exit 1
fi

# Start the Flask backend
cd /backend || { echo "Backend directory not found"; exit 1; }

echo "Starting Flask backend"
# Security: Check Python version
python3 --version

# Security: Verify requirements are installed
echo "Verifying dependencies..."
pip3 list | grep -E 'flask|requests|beautifulsoup4'
if [ $? -ne 0 ]; then
    echo "ERROR: Critical dependencies missing!"
    exit 1
fi

# Start Flask app with health check
echo "Starting Flask app..."
python3 app.py --host 0.0.0.0 --port 8001 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
# Implement a proper health check with timeout
MAX_RETRIES=30
RETRY=0
while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8001/health 2>/dev/null | grep -q "ok"; then
        echo "Backend started successfully!"
        break
    fi
    
    # Check if process is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "ERROR: Backend process died during startup!"
        exit 1
    fi
    
    echo "Waiting for backend to become available... ($RETRY/$MAX_RETRIES)"
    sleep 1
    RETRY=$((RETRY+1))
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "ERROR: Backend failed to start within the timeout period!"
    kill $BACKEND_PID
    exit 1
fi

# Start Nginx with configtest
echo "Starting Nginx..."
nginx -t
if [ $? -ne 0 ]; then
    echo "ERROR: Nginx configuration test failed!"
    kill $BACKEND_PID
    exit 1
fi

nginx -g 'daemon off;' &
NGINX_PID=$!

# Handle termination signals
trap 'echo "Received shutdown signal"; kill $BACKEND_PID $NGINX_PID; exit 0' SIGTERM SIGINT

echo "CyberGuard AI is now running!"

# Monitor processes
while kill -0 $BACKEND_PID 2>/dev/null && kill -0 $NGINX_PID 2>/dev/null; do
    sleep 5
done

# If we get here, one of the processes died
if kill -0 $BACKEND_PID 2>/dev/null; then
    echo "ERROR: Nginx died, shutting down backend..."
    kill $BACKEND_PID
else
    echo "ERROR: Backend died, shutting down nginx..."
    kill $NGINX_PID
fi

exit 1
