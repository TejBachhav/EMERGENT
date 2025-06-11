#!/usr/bin/env python3
"""
CyberGuard AI - Simple Backend with GPT4All Mistral Instruct 7B Q4
A cybersecurity-focused chatbot using local GPT4All model
"""

from flask import Flask, request, jsonify, session, Response
from datetime import datetime
import json
import os
import logging
from gpt4all import GPT4All
import torch

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "cyberguard-secret-key-2025"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cyberguard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Check GPU availability
def is_gpu_available():
    return torch.cuda.is_available()

# Load GPT4All Mistral model
model_path = "C:/Users/Tej Bachhav/AppData/Local/nomic.ai/GPT4All"
model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"

try:
    llm = GPT4All(
        model_name,
        model_path=model_path,
        allow_download=False,
        device="cuda" if is_gpu_available() else "cpu"
    )
    logger.info(f"[SUCCESS] GPT4All model loaded: {model_name}")
    logger.info(f"[DEVICE] Device: {'GPU (CUDA)' if is_gpu_available() else 'CPU'}")
except Exception as e:
    logger.error(f"[ERROR] Failed to load GPT4All model: {e}")
    llm = None

# ============================================================================
# VULNERABILITY DATABASE
# ============================================================================

VULNERABILITIES = {
    "sql_injection": {
        "name": "SQL Injection",
        "description": "SQL injection occurs when untrusted data is sent to an interpreter as part of a command or query, allowing attackers to execute unintended commands.",
        "severity": "Critical",
        "cvss_score": 9.3,
        "example": "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "secure_example": "SELECT * FROM users WHERE id = ?",
        "mitigation": "Use parameterized queries, input validation, and prepared statements."
    },
    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "XSS flaws occur when an application includes untrusted data in a new web page without proper validation or escaping.",
        "severity": "High",
        "cvss_score": 7.5,
        "example": "document.innerHTML = userInput",
        "secure_example": "document.textContent = userInput",
        "mitigation": "Validate input, encode output, use Content Security Policy (CSP)."
    },
    "csrf": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "description": "CSRF forces an end user to execute unwanted actions on a web application in which they're currently authenticated.",
        "severity": "Medium",
        "cvss_score": 6.5,
        "example": "<img src='http://bank.com/transfer?to=attacker&amount=1000'>",
        "secure_example": "<form><input type='hidden' name='csrf_token' value='{{token}}'>",
        "mitigation": "Use anti-CSRF tokens, SameSite cookies, and verify referrer headers."
    },
    "buffer_overflow": {
        "name": "Buffer Overflow",
        "description": "Buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code.",
        "severity": "Critical",
        "cvss_score": 8.8,
        "example": "char buffer[10]; strcpy(buffer, user_input);",
        "secure_example": "char buffer[10]; strncpy(buffer, user_input, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\\0';",
        "mitigation": "Use safe string functions and always validate input length."
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_vulnerability_type(query):
    """Detect if query is about a specific vulnerability"""
    query_lower = query.lower().replace('-', ' ').replace('_', ' ')
    
    if any(keyword in query_lower for keyword in ['sql', 'injection', 'sqli', 'database']):
        return 'sql_injection'
    elif any(keyword in query_lower for keyword in ['xss', 'cross site scripting', 'script injection']):
        return 'xss'
    elif any(keyword in query_lower for keyword in ['csrf', 'cross site request forgery']):
        return 'csrf'
    elif any(keyword in query_lower for keyword in ['buffer', 'overflow', 'memory']):
        return 'buffer_overflow'
    else:
        return 'general'

def get_risk_category(cvss_score):
    """Get risk category based on CVSS score"""
    if cvss_score >= 9.0:
        return "🔴 Critical Risk"
    elif cvss_score >= 7.0:
        return "🟠 High Risk"
    elif cvss_score >= 4.0:
        return "🟡 Medium Risk"
    else:
        return "🟢 Low Risk"

def build_security_prompt(query, vuln_type=None):
    """Build a security-focused prompt for the model"""
    
    base_context = """You are CyberGuard AI, an expert cybersecurity assistant. You help developers understand security vulnerabilities, provide secure coding practices, and analyze potential threats.

Always provide:
- Clear explanations of security concepts
- Practical code examples (vulnerable and secure versions)
- Specific mitigation strategies
- Best practices and recommendations

Keep responses professional, informative, and actionable."""
    
    if vuln_type and vuln_type in VULNERABILITIES:
        vuln = VULNERABILITIES[vuln_type]
        context = f"""
{base_context}

Relevant vulnerability context:
- Vulnerability: {vuln['name']}
- Severity: {vuln['severity']} (CVSS: {vuln['cvss_score']})
- Description: {vuln['description']}
- Example vulnerable code: {vuln['example']}
- Secure implementation: {vuln['secure_example']}
- Mitigation: {vuln['mitigation']}
"""
    else:
        context = base_context
    
    return f"""{context}

User Query: {query}

Response:"""

def generate_response_with_gpt4all(prompt, max_tokens=1000):
    """Generate response using GPT4All model"""
    if not llm:
        return None
    
    try:
        response = llm.generate(
            prompt,
            max_tokens=max_tokens,
            temp=0.7,
            top_p=0.9,
            top_k=40
        )
        return response.strip()
    except Exception as e:
        logger.error(f"Error generating response with GPT4All: {e}")
        return None

def generate_vulnerability_response(vuln_type, query):
    """Generate a response for a specific vulnerability type"""
    if vuln_type not in VULNERABILITIES:
        return None
    
    vuln = VULNERABILITIES[vuln_type]
    
    # Use GPT4All for detailed response
    prompt = build_security_prompt(query, vuln_type)
    ai_response = generate_response_with_gpt4all(prompt)
    
    if ai_response:
        response = f"""🛡️ **{vuln['name']} Security Analysis**

{ai_response}

---

**📊 Technical Details:**
- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)
- **Risk Category:** {get_risk_category(vuln['cvss_score'])}

**🚨 Vulnerable Code Example:**
```sql
{vuln['example']}
```

**✅ Secure Implementation:**
```sql
{vuln['secure_example']}
```

**🔧 Primary Mitigation:**
{vuln['mitigation']}

**🛠️ Additional Security Measures:**
• Implement input validation and sanitization
• Use parameterized queries and prepared statements
• Apply principle of least privilege for database access
• Enable SQL injection detection tools
• Regular security code reviews and penetration testing

**📚 References:**
• OWASP Top 10 Security Risks
• CWE-89: Improper Neutralization of Special Elements
• NIST Cybersecurity Framework"""
    else:
        # Fallback response if model fails
        response = f"""🛡️ **{vuln['name']} Security Analysis**

**📝 Description:**
{vuln['description']}

**📊 Risk Assessment:**
- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)
- **Risk Category:** {get_risk_category(vuln['cvss_score'])}

**🚨 Vulnerable Code Example:**
```sql
{vuln['example']}
```

**✅ Secure Implementation:**
```sql
{vuln['secure_example']}
```

**🔧 Mitigation Strategies:**
{vuln['mitigation']}

**🛠️ Additional Security Recommendations:**
• Implement comprehensive input validation and sanitization
• Use security-focused development frameworks and libraries
• Conduct regular security testing and code reviews
• Follow OWASP security guidelines and best practices
• Keep dependencies updated and monitor for vulnerabilities
• Implement proper error handling without information disclosure
• Use Web Application Firewalls (WAF) as additional protection

**📚 Security Resources:**
• OWASP Top 10 Security Risks
• SANS Top 25 Most Dangerous Software Errors
• NIST Cybersecurity Framework Guidelines"""
    
    return response

def generate_general_response(query):
    """Generate a general cybersecurity response using GPT4All"""
    prompt = build_security_prompt(query)
    ai_response = generate_response_with_gpt4all(prompt, max_tokens=1500)
    
    if ai_response:
        return f"🔒 **CyberGuard AI Security Guidance**\n\n{ai_response}\n\n---\n\n**🛡️ Security Best Practices Reminder:**\n• Keep all systems and dependencies updated\n• Implement defense-in-depth security strategies\n• Regular security assessments and penetration testing\n• Follow OWASP and NIST cybersecurity guidelines"
    else:
        return """🔒 **CyberGuard AI Security Guidance**

I'm experiencing technical difficulties with my AI processing capabilities right now. However, I can still help with common cybersecurity topics:

**🔐 General Security Best Practices:**
• **Authentication**: Use multi-factor authentication (MFA)
• **Authorization**: Implement principle of least privilege
• **Input Validation**: Sanitize and validate all user inputs
• **Encryption**: Use TLS for data in transit, AES for data at rest
• **Updates**: Keep all software and dependencies current
• **Monitoring**: Implement comprehensive logging and alerting

**🎯 Common Vulnerability Categories (OWASP Top 10):**
1. **Injection Flaws** (SQL, NoSQL, OS injection)
2. **Broken Authentication** and session management
3. **Sensitive Data Exposure**
4. **XML External Entities (XXE)**
5. **Broken Access Control**
6. **Security Misconfiguration**
7. **Cross-Site Scripting (XSS)**
8. **Insecure Deserialization**
9. **Using Components with Known Vulnerabilities**
10. **Insufficient Logging & Monitoring**

**🔍 Security Assessment Areas:**
• Code review and static analysis
• Dynamic application security testing (DAST)
• Interactive application security testing (IAST)
• Software composition analysis (SCA)
• Infrastructure security scanning

**📚 Recommended Resources:**
• OWASP Testing Guide
• NIST Cybersecurity Framework
• SANS Top 25 Software Errors
• CVE Database and NVD

Please try your query again, or ask about a specific vulnerability type like SQL injection, XSS, or CSRF."""

def truncate_response(response, max_length=4000):
    """Truncate response if too long to prevent session overflow"""
    if len(response) <= max_length:
        return response
    
    truncated = response[:max_length]
    # Try to cut at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.8:  # If we can cut at 80% or more
        truncated = truncated[:last_period + 1]
    
    return truncated

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """Health check and service info"""
    return jsonify({
        "service": "CyberGuard AI",
        "version": "3.0.0",
        "status": "operational",
        "model": "GPT4All Mistral Instruct 7B Q4",
        "device": "GPU (CUDA)" if is_gpu_available() else "CPU",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "endpoints": {
            "chat": "/stream_chat",
            "health": "/api/health",
            "vulnerabilities": "/api/vulnerabilities"
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    model_status = "operational" if llm else "not_loaded"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "api_server": "operational",
            "gpt4all_model": model_status,
            "vulnerability_db": "operational"
        },
        "model_info": {
            "name": "Mistral Instruct 7B Q4",
            "device": "GPU (CUDA)" if is_gpu_available() else "CPU",
            "loaded": llm is not None
        },
        "uptime": "99.9%",
        "threat_level": "LOW"
    })

@app.route('/api/vulnerabilities', methods=['GET'])
def list_vulnerabilities():
    """List available vulnerability information"""
    vuln_list = []
    for key, vuln in VULNERABILITIES.items():
        vuln_list.append({
            "id": key,
            "name": vuln["name"],
            "severity": vuln["severity"],
            "cvss_score": vuln["cvss_score"]
        })
    
    return jsonify({
        "vulnerabilities": vuln_list,
        "total": len(vuln_list)
    })

@app.route('/stream_chat', methods=['POST'])
def stream_chat():
    """Main chat endpoint for cybersecurity queries with streaming response"""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        
        # Log the query
        logger.info(f"Processing query: {user_message[:100]}...")
        
        # Detect vulnerability type
        vuln_type = detect_vulnerability_type(user_message)
        
        # Generate response
        if vuln_type != 'general':
            response = generate_vulnerability_response(vuln_type, user_message)
            source = "vulnerability_database + gpt4all"
        else:
            response = generate_general_response(user_message)
            source = "gpt4all" if llm else "fallback"
        
        # Truncate response if too long
        response = truncate_response(response)
        
        # Manage conversation history
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Add to conversation history
        session['conversation'].append({"role": "user", "content": user_message})
        session['conversation'].append({"role": "assistant", "content": response})
        
        # Keep only last 3 exchanges to prevent session overflow
        session['conversation'] = session['conversation'][-6:]  # 6 = 3 exchanges * 2 messages each
        
        # Log response stats
        logger.info(f"Response generated: {len(response)} chars, source: {source}")
        
        # Log query for analytics
        log_file = "query_log.jsonl"
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        
        with open(log_file, "a", encoding='utf-8') as f:
            f.write(json.dumps({
                "timestamp": datetime.now().isoformat(),
                "input": user_message,
                "classification": vuln_type,
                "source": source,
                "response_length": len(response)
            }, ensure_ascii=False) + "\n")
        
        # Stream the response character by character
        def generate_stream():
            for char in response:
                yield char
        
        return Response(generate_stream(), mimetype="text/plain")
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        error_response = "[SYSTEM ERROR] I'm experiencing technical difficulties. Please try again in a moment."
        
        def error_stream():
            for char in error_response:
                yield char
        
        return Response(error_stream(), mimetype="text/plain")

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get conversation statistics"""
    conversation_count = len(session.get('conversation', [])) // 2  # Divide by 2 (user + bot pairs)
    
    return jsonify({
        "uptime": "99.9%",
        "threat_level": "LOW",
        "messages": conversation_count,
        "active_scans": 0,
        "vulnerabilities_detected": conversation_count,
        "security_score": 95,
        "model_status": "operational" if llm else "offline"
    })

# ============================================================================
# CORS HANDLERS
# ============================================================================

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = Response()
        # Get the origin from the request header
        origin = request.headers.get('Origin')
        # Allow specific origins instead of wildcard when using credentials
        allowed_origins = ["http://localhost:3000", "http://localhost:5000", "http://localhost:9000"]
        if origin in allowed_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    origin = request.headers.get('Origin')
    allowed_origins = ["http://localhost:3000", "http://localhost:5000", "http://localhost:9000"]
    if origin in allowed_origins:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested resource does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("[CYBERGUARD AI] Starting Security Backend with GPT4All")
    print("=" * 60)
    print(f"[SERVER] Server: http://localhost:9000")
    print(f"[CHAT] Chat Endpoint: /stream_chat")
    print(f"[HEALTH] Health Check: /api/health")
    print(f"[VULNS] Vulnerabilities: /api/vulnerabilities")
    print(f"[STATS] Stats: /api/stats")
    print("=" * 60)
    
    if not llm:
        print("[WARNING] GPT4All model not loaded - using fallback responses")
        print("[PATH] Expected model path:", model_path)
        print("[FILE] Expected model file:", model_name)
    else:
        print(f"[SUCCESS] GPT4All model loaded: {model_name}")
        print(f"[DEVICE] Device: {'GPU (CUDA)' if is_gpu_available() else 'CPU'}")
    
    print("[READY] Ready for cybersecurity queries!")
    print()
    
    app.run(
        host='0.0.0.0',
        port=9000,
        debug=True
    )
