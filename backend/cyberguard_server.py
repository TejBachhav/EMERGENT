from flask import Flask, render_template, request, session, Response, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
import sys

app = Flask(__name__)
app.secret_key = "supersecurekey"

# Enable CORS with credentials support for session management
CORS(app, supports_credentials=True, origins=["*"])

log_file = "/app/backend/query_log.jsonl"

# Mock vulnerability database for demo (simulating your FAISS + GPT4All setup)
MOCK_VULNERABILITIES = {
    "sql_injection": {
        "name": "SQL Injection",
        "description": "SQL injection is a code injection technique that might destroy your database. It occurs when user input is not properly sanitized before being included in SQL queries.",
        "examples": [{
            "code": "SELECT * FROM users WHERE username = '" + "' + username + '" + "' AND password = '" + "' + password + '" + "'"
        }],
        "remediations": [{
            "code": "SELECT * FROM users WHERE username = ? AND password = ?"
        }],
        "patch": "Use parameterized queries or prepared statements to prevent SQL injection attacks."
    },
    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "XSS attacks enable attackers to inject client-side scripts into web pages viewed by other users.",
        "examples": [{
            "code": "document.innerHTML = userInput;"
        }],
        "remediations": [{
            "code": "document.textContent = userInput; // or use proper sanitization"
        }],
        "patch": "Always sanitize user input and use proper encoding when displaying user data."
    },
    "csrf": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "description": "CSRF attacks trick the victim into performing actions they didn't intend to perform.",
        "examples": [{
            "code": "<form action='/transfer' method='POST'><input name='amount' value='1000'></form>"
        }],
        "remediations": [{
            "code": "<form action='/transfer' method='POST'><input type='hidden' name='csrf_token' value='{{csrf_token}}'></form>"
        }],
        "patch": "Implement CSRF tokens and validate them on all state-changing operations."
    },
    "buffer_overflow": {
        "name": "Buffer Overflow",
        "description": "Buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code.",
        "examples": [{
            "code": "char buffer[10];\\nstrcpy(buffer, user_input); // Dangerous if user_input > 10 chars"
        }],
        "remediations": [{
            "code": "char buffer[10];\\nstrncpy(buffer, user_input, sizeof(buffer)-1);\\nbuffer[sizeof(buffer)-1] = '\\\\0';"
        }],
        "patch": "Use safe string functions and always validate input length."
    }
}

def detect_vulnerability_type(query):
    """Simple keyword-based detection for demo purposes"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ['sql', 'injection', 'database', 'query']):
        return 'sql_injection'
    elif any(keyword in query_lower for keyword in ['xss', 'script', 'javascript', 'dom']):
        return 'xss'
    elif any(keyword in query_lower for keyword in ['csrf', 'cross-site', 'request forgery']):
        return 'csrf'
    elif any(keyword in query_lower for keyword in ['buffer', 'overflow', 'memory']):
        return 'buffer_overflow'
    else:
        return 'general'

def generate_security_response(query, vuln_type):
    """Generate a security-focused response"""
    
    if vuln_type in MOCK_VULNERABILITIES:
        vuln = MOCK_VULNERABILITIES[vuln_type]
        response = f"""🛡️ **{vuln['name']} Analysis**

**Description:**
{vuln['description']}

**Vulnerable Code Example:**
```javascript
{vuln['examples'][0]['code']}
```

**Secure Implementation:**
```javascript
{vuln['remediations'][0]['code']}
```

**Security Patch:**
{vuln['patch']}

**Additional Security Recommendations:**
• Always validate and sanitize user input
• Implement proper authentication and authorization
• Use security headers (CSP, X-Frame-Options, etc.)
• Regular security audits and penetration testing
• Keep dependencies up to date
• Follow OWASP guidelines for secure coding

**Next Steps:**
• Review your codebase for similar patterns
• Implement automated security testing
• Set up continuous security monitoring

Would you like me to analyze specific code or explain other security vulnerabilities?"""
    else:
        # General security response
        response = f"""🛡️ **CyberGuard AI Security Analysis**

Thank you for your security question: "{query}"

**Security Assessment Complete**

**General Security Best Practices:**

• **Input Validation**: Always validate and sanitize user input before processing
• **Authentication**: Implement multi-factor authentication where possible
• **Authorization**: Use principle of least privilege for access controls
• **Encryption**: Encrypt sensitive data both in transit (TLS) and at rest
• **Logging**: Implement comprehensive security event logging
• **Updates**: Maintain up-to-date systems and dependencies

**OWASP Top 10 Security Risks:**
1. **Injection Flaws** - SQL, NoSQL, OS command injection
2. **Broken Authentication** - Session management vulnerabilities
3. **Sensitive Data Exposure** - Insufficient protection of sensitive data
4. **XML External Entities (XXE)** - Poorly configured XML processors
5. **Broken Access Control** - Improperly enforced restrictions
6. **Security Misconfiguration** - Insecure default configurations
7. **Cross-Site Scripting (XSS)** - Untrusted data in web pages
8. **Insecure Deserialization** - Remote code execution flaws
9. **Using Components with Known Vulnerabilities** - Outdated libraries
10. **Insufficient Logging & Monitoring** - Inadequate incident detection

**Code Review Checklist:**
✅ Input validation and sanitization
✅ Output encoding
✅ Authentication mechanisms
✅ Session management
✅ Access controls
✅ Error handling
✅ Logging implementation

**Need Specific Help?**
Please provide:
• Code snippets for security review
• Architecture diagrams for threat modeling
• Specific vulnerability concerns
• Compliance requirements (PCI DSS, HIPAA, etc.)

How can I assist you further with your cybersecurity needs?"""
    
    return response

# === Flask Routes ===
@app.route("/")
def index():
    return jsonify({
        "service": "CyberGuard AI Security Backend",
        "status": "running",
        "version": "1.0.0",
        "endpoints": ["/stream_chat", "/api/health", "/api/stats"]
    })

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "CyberGuard AI",
        "uptime": "99.9%",
        "threat_level": "LOW",
        "security_modules": ["vulnerability_scanner", "code_analyzer", "threat_detector"]
    })

@app.route("/stream_chat", methods=["POST"])
def stream_chat():
    try:
        data = request.get_json()
        user_input = data.get("message", "").strip()
        
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
        
        print(f"Received message: {user_input}")
        
        # Get or initialize conversation history
        if 'conversation' not in session:
            session['conversation'] = []
        
        history = session.get("conversation", [])
        
        # Detect vulnerability type
        vuln_type = detect_vulnerability_type(user_input)
        print(f"Detected vulnerability type: {vuln_type}")
        
        # Generate response
        full_response = generate_security_response(user_input, vuln_type)
        
        # Update session history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": full_response})
        session["conversation"] = history
        
        # Log the query
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "a") as f_log:
            f_log.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "classification": vuln_type,
                "session_id": session.get('session_id', 'unknown')
            }) + "\n")
        
        print(f"Sending response: {len(full_response)} characters")
        
        # Return streaming response
        def generate_stream():
            for char in full_response:
                yield char
        
        return Response(generate_stream(), mimetype="text/plain")
    
    except Exception as e:
        print(f"Error in stream_chat: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Internal server error"}), 500

@app.route("/api/stats", methods=["GET"])
def get_stats():
    conversation_count = len(session.get('conversation', [])) // 2  # Divide by 2 (user + bot pairs)
    return jsonify({
        "uptime": "99.9%",
        "threat_level": "LOW",
        "messages": conversation_count,
        "active_scans": 0,
        "vulnerabilities_detected": conversation_count,
        "security_score": 95
    })

# Add this route to handle OPTIONS requests for CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        response.headers.add('Access-Control-Allow-Credentials', "true")
        return response

if __name__ == "__main__":
    print("🛡️ Starting CyberGuard AI Security Backend...")
    print("📡 Backend will be available at: http://localhost:5000")
    print("🔗 Frontend connects to: /stream_chat endpoint")
    print("✅ CORS enabled for frontend integration")
    print("📝 Query logs saved to:", log_file)
    print("🔍 Security modules: vulnerability_scanner, code_analyzer, threat_detector")
    
    app.run(debug=True, host="0.0.0.0", port=5000)