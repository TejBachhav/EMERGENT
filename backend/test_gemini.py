#!/usr/bin/env python3
"""
Simple test script to verify Gemini API integration
"""
import json
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("❌ GEMINI_API_KEY environment variable is not set.")
    exit(1)

def test_gemini_api(query="What is SQL injection?"):
    """Test Gemini API with a simple query"""
    print(f"🧪 Testing Gemini API with query: '{query}'")
    
    gemini_api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={gemini_api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"You are a cybersecurity expert. Provide a clear, formatted response about: {query}. Include practical security advice and code examples if relevant."
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(gemini_api_url, json=payload, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        print("✅ Raw API Response:")
        print(json.dumps(data, indent=2)[:500] + "...")
        
        # Extract the actual text content
        if 'candidates' in data and len(data['candidates']) > 0:
            if 'content' in data['candidates'][0] and 'parts' in data['candidates'][0]['content']:
                extracted_text = data['candidates'][0]['content']['parts'][0]['text']
                print("\n✅ Extracted Text:")
                print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
                return extracted_text
        
        print("❌ Could not extract text from response")
        return None
        
    except requests.RequestException as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    test_gemini_api()
