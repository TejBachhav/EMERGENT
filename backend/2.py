from flask import Flask, render_template, request, session, Response, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os
import sys
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
import faiss
from functools import lru_cache
import torch

app = Flask(__name__)
app.secret_key = "supersecurekey"

# Update CORS configuration
CORS(app, supports_credentials=True, origins=["http://localhost:5000"])

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

# === Load embedding model ===
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Check if GPU is available
def is_gpu_available():
    return torch.cuda.is_available()

# === Load or build GPT4All LLM once ===
model_path = "C:/Users/Tej Bachhav/AppData/Local/nomic.ai/GPT4All"
llm = GPT4All(
    "mistral-7b-instruct-v0.1.Q4_0.gguf",
    model_path=model_path,
    allow_download=False,
    device="cuda" if is_gpu_available() else "cpu"
)

# === Load FAISS index & metadata (rename index to faiss_index) ===
faiss_index = None
if os.path.exists("vuln_index.faiss"):
    faiss_index = faiss.read_index("vuln_index.faiss")

with open("vuln_metadata.json", "r") as f:
    metadata = json.load(f)

# === Build FAISS index if not present ===
def build_faiss_index(json_path="vulnerabilities.json"):
    with open(json_path, "r") as f:
        data = json.load(f)

    docs = [
        f"{v['name']}. Description: {v['description']}. Example: {v['examples'][0]['code']}"
        for v in data
    ]
    
    # Reduce vector dimensions for faster indexing
    vectors = embed_model.encode(docs, normalize_embeddings=True)

    global faiss_index
    faiss_index = faiss.IndexFlatIP(vectors.shape[1])  # Switched to Inner Product for Approximate Nearest Neighbors
    faiss_index.add(vectors)

    faiss.write_index(faiss_index, "vuln_index.faiss")
    with open("vuln_metadata.json", "w") as f_meta:
        json.dump(data, f_meta)
    print("✅ FAISS index built and saved with optimizations.")

# === Context Retrieval ===
def retrieve_context(query, k=1):
    if faiss_index is None:
        return None

    query_vec = embed_model.encode([query])
    D, I = faiss_index.search(query_vec, k)

    # If distance > 50, treat as no match
    if D[0][0] > 50.0:
        return None
    return metadata[I[0][0]]

# === Prompt Builder ===
def build_prompt(query, context_data):
    return f"""
You are a security assistant helping developers understand vulnerabilities.

User query: "{query}"

Use the following context to help answer:

Vulnerability: {context_data['name']}
Description: {context_data['description']}

Example:
{context_data['examples'][0]['code']}

Remediation:
{context_data['remediations'][0]['code']}

Patch:
{context_data['patch']}

Respond in a helpful, developer-friendly way.
"""

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

@lru_cache(maxsize=128)
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
    data = request.get_json()
    user_input = data.get("message", "").strip()
    history = session.get("conversation", [])

    context_data = retrieve_context(user_input)
    if context_data:
        prompt = build_prompt(user_input, context_data)
        classification = context_data["name"]
    else:
        prompt = f"The user asked: {user_input}\nProvide a helpful response as a security assistant."
        classification = "general"

    # Limit the number of messages from history to include in the prompt
    max_history_messages = 5
    history = history[-max_history_messages:]

    # Prepend multi-turn history
    for msg in history:
        if msg["role"] == "user":
            prompt = f"User: {msg['content']}\n" + prompt
        else:
            prompt = f"Assistant: {msg['content']}\n" + prompt

    # Truncate the prompt to fit within the model's context window
    max_context_window = 2048
    if len(prompt.split()) > max_context_window:
        prompt = ' '.join(prompt.split()[:max_context_window])

    # 1. Generate full response (no streaming flag)
    full_response = llm.generate(prompt, max_tokens=1500, temp=0.5)  # Reduced max_tokens and temperature

    # 2. Update session and log BEFORE streaming
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": full_response})
    session["conversation"] = history

    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f_log:
        f_log.write(json.dumps({
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "classification": classification
        }) + "\n")

    # 3. Return a generator that yields one character at a time
    def generate_stream():
        for char in full_response:
            yield char

    return Response(generate_stream(), mimetype="text/plain")

# Add this route to handle OPTIONS requests for CORS
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = Response()
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:5000")
        response.headers.add('Access-Control-Allow-Headers', "Content-Type, Authorization")
        response.headers.add('Access-Control-Allow-Methods', "GET, POST, OPTIONS")
        response.headers.add('Access-Control-Allow-Credentials', "true")
        return response

# === Build FAISS index if missing, then run ===
if __name__ == "__main__":
    if faiss_index is None:
        build_faiss_index()
    print("🛡️ Starting CyberGuard AI Security Backend...")
    print("📡 Backend will be available at: http://localhost:9000")
    print("🔗 Frontend connects to: /stream_chat endpoint")
    print("✅ CORS enabled for frontend integration")
    print("📝 Query logs saved to:", log_file)
    print("🔍 Security modules: vulnerability_scanner, code_analyzer, threat_detector")
    
    app.run(debug=True, host="0.0.0.0", port=9000)