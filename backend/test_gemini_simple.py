#!/usr/bin/env python3
"""
Simple test script to verify Gemini API integration
"""

import os
import sys
import requests
import json

def test_gemini_api():
    """Test if Gemini API key is set and working"""
    
    print("🔍 Testing Gemini API Integration...")
    print("=" * 50)
    
    # Check if API key is set
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY environment variable not set")
        print("\n💡 To fix this:")
        print("1. Get API key from: https://makersuite.google.com/app/apikey")
        print("2. Set environment variable:")
        print('   $env:GEMINI_API_KEY="your-api-key-here"')
        return False
    
    print(f"✅ API key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Test API call
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "What is SQL injection? Provide a brief explanation."
                    }
                ]
            }
        ]
    }
    
    try:
        print("🚀 Making test API call...")
        response = requests.post(url, headers=headers, json=payload, timeout=15)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            response_json = response.json()
            
            if response_json.get('candidates') and \
               len(response_json['candidates']) > 0 and \
               response_json['candidates'][0].get('content') and \
               response_json['candidates'][0]['content'].get('parts') and \
               len(response_json['candidates'][0]['content']['parts']) > 0 and \
               response_json['candidates'][0]['content']['parts'][0].get('text'):
                
                context = response_json['candidates'][0]['content']['parts'][0]['text']
                print("✅ Gemini API is working!")
                print(f"📝 Response preview: {context[:200]}...")
                return True
            else:
                print("❌ Unexpected response format")
                print(f"Response: {response_json}")
                return False
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error making API call: {e}")
        return False

if __name__ == "__main__":
    success = test_gemini_api()
    
    if success:
        print("\n🎉 Gemini API integration is ready!")
        print("Your Flask server will now use Gemini for context retrieval.")
    else:
        print("\n🔧 Please fix the issues above and try again.")
    
    sys.exit(0 if success else 1)
