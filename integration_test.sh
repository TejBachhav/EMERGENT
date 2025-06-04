#!/bin/bash

echo "🛡️ Testing CyberGuard AI Integration"
echo "======================================"

echo ""
echo "1. Testing Frontend Availability..."
FRONTEND_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000)
if [ "$FRONTEND_STATUS" = "200" ]; then
    echo "✅ Frontend running on localhost:3000"
else
    echo "❌ Frontend not accessible: HTTP $FRONTEND_STATUS"
fi

echo ""
echo "2. Testing Backend Health..."
BACKEND_HEALTH=$(curl -s http://localhost:5000/api/health | jq -r '.status' 2>/dev/null)
if [ "$BACKEND_HEALTH" = "healthy" ]; then
    echo "✅ Backend healthy on localhost:5000"
    curl -s http://localhost:5000/api/health | jq '.' 2>/dev/null || echo "Backend response: $(curl -s http://localhost:5000/api/health)"
else
    echo "❌ Backend not healthy"
fi

echo ""
echo "3. Testing Chat Endpoint..."
CHAT_TEST=$(curl -s -X POST http://localhost:5000/stream_chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}' | head -c 100)
if [ -n "$CHAT_TEST" ]; then
    echo "✅ Chat endpoint responding"
    echo "Response preview: $CHAT_TEST..."
else
    echo "❌ Chat endpoint not responding"
fi

echo ""
echo "4. Environment Configuration..."
echo "REACT_APP_BACKEND_URL in frontend/.env:"
cat /app/frontend/.env | grep REACT_APP_BACKEND_URL

echo ""
echo "5. Integration Test Summary..."
if [ "$FRONTEND_STATUS" = "200" ] && [ "$BACKEND_HEALTH" = "healthy" ] && [ -n "$CHAT_TEST" ]; then
    echo "🎉 INTEGRATION SUCCESS: All components ready!"
    echo ""
    echo "✨ Your Beautiful Security Chat Interface is LIVE!"
    echo ""
    echo "🌟 Features Available:"
    echo "   • Beautiful dark cybersecurity theme"
    echo "   • Streaming chat responses"
    echo "   • Code syntax highlighting"
    echo "   • Message reactions system"
    echo "   • Quick security prompts"
    echo "   • Animated background effects"
    echo "   • Professional security analysis"
    echo ""
    echo "🔗 Access your chat interface at: http://localhost:3000"
    echo "🛡️ Backend API available at: http://localhost:5000"
else
    echo "⚠️  Some components need attention"
fi

echo ""
echo "======================================"