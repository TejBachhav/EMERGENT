"""
CyberGuard AI - Simple Backend with GPT4All Mistral Instruct 7B Q4
A cybersecurity-focused chatbot using local GPT4All model
"""

from flask import Flask, request, jsonify, session, Response
from datetime import datetime
import json
import os
import logging
import requests # Added import
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
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:H/UI:N/S:U/C:H/I:H/A:N",
        "example": "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "secure_example": "SELECT * FROM users WHERE id = ?",
        "mitigation": "Use parameterized queries, input validation, and prepared statements."
    },    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "XSS flaws occur when an application includes untrusted data in a new web page without proper validation or escaping.",
        "severity": "High",
        "cvss_score": 7.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N",
        "example": "document.innerHTML = userInput",
        "secure_example": "document.textContent = userInput",
        "mitigation": "Validate input, encode output, use Content Security Policy (CSP)."
    },    "csrf": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "description": "CSRF forces an end user to execute unwanted actions on a web application in which they're currently authenticated.",
        "severity": "Medium",
        "cvss_score": 6.5,
        "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:N",
        "example": "<img src='http://bank.com/transfer?to=attacker&amount=1000'>",
        "secure_example": "<form><input type='hidden' name='csrf_token' value='{{token}}'>",
        "mitigation": "Use anti-CSRF tokens, SameSite cookies, and verify referrer headers."
    },    "buffer_overflow": {
        "name": "Buffer Overflow",
        "description": "Buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code.",
        "severity": "Critical",
        "cvss_score": 8.8,
        "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "example": "char buffer[10]; strcpy(buffer, user_input);",
        "secure_example": "char buffer[10]; strncpy(buffer, user_input, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\\0';",
        "mitigation": "Use safe string functions and always validate input length."
    }
}

# ============================================================================
# GEMINI API CONTEXT RETRIEVAL
# ============================================================================

class GeminiAPI:
    def __init__(self):
        self.api_call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
    def get_context(self, query: str) -> str:
        """
        Retrieves short, relevant context from the Gemini API.
        """
        import time
        start_time = time.time()
        self.api_call_count += 1
        
        logger.info(f"[GEMINI] Starting API call #{self.api_call_count} for query: {query[:100]}{'...' if len(query) > 100 else ''}")
        
        from dotenv import load_dotenv
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.failed_calls += 1
            logger.error("[GEMINI] GEMINI_API_KEY not found in environment variables.")
            return "Gemini Insight: API Key not configured. Unable to fetch real-time context."

        # Corrected endpoint from the user's cURL command
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        logger.debug(f"[GEMINI] Using endpoint: {url.split('?')[0]}...")  # Log URL without API key
        headers = {
            'Content-Type': 'application/json'
        }
        
        # Payload structure based on the cURL command
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": query # Use the user's query as the input text
                        }
                    ]
                }
            ]
        }
        
        logger.debug(f"[GEMINI] Request payload size: {len(str(payload))} characters")
        
        try:
            logger.info("[GEMINI] Making POST request to Gemini API...")
            response = requests.post(url, headers=headers, json=payload, timeout=15) # Added timeout
            logger.info(f"[GEMINI] Response received - Status Code: {response.status_code}")
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            
            response_json = response.json()
            logger.debug(f"[GEMINI] Response JSON keys: {list(response_json.keys())}")
              # Navigate the JSON structure to get the text
            # Based on typical Gemini API responses, the path might be:
            # response_json['candidates'][0]['content']['parts'][0]['text']
            if response_json.get('candidates') and \
               len(response_json['candidates']) > 0 and \
               response_json['candidates'][0].get('content') and \
               response_json['candidates'][0]['content'].get('parts') and \
               len(response_json['candidates'][0]['content']['parts']) > 0 and \
               response_json['candidates'][0]['content']['parts'][0].get('text'):
                context = response_json['candidates'][0]['content']['parts'][0]['text']
                elapsed_time = time.time() - start_time
                self.successful_calls += 1
                logger.info(f"[GEMINI] Successfully retrieved context from Gemini API - Length: {len(context)} characters, Time: {elapsed_time:.2f}s")
                logger.debug(f"[GEMINI] Context preview: {context[:200]}{'...' if len(context) > 200 else ''}")
                logger.info(f"[GEMINI] API Stats - Total: {self.api_call_count}, Success: {self.successful_calls}, Failed: {self.failed_calls}")
                return f"Gemini Insight: {context}"
            else:
                self.failed_calls += 1
                elapsed_time = time.time() - start_time
                logger.warning(f"[GEMINI] API response did not contain expected text structure - Time: {elapsed_time:.2f}s")
                logger.warning(f"[GEMINI] Response keys: {list(response_json.keys())}")
                logger.debug(f"[GEMINI] Full response for debugging: {response_json}")
                return "Gemini Insight: Received an unexpected response format from the context API."

        except requests.exceptions.HTTPError as http_err:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] HTTP error occurred: {http_err} - Status Code: {response.status_code}, Time: {elapsed_time:.2f}s")
            logger.error(f"[GEMINI] Response content: {response.text[:500]}{'...' if len(response.text) > 500 else ''}")
            return f"Gemini Insight: Error communicating with context API (HTTP {response.status_code})."
        except requests.exceptions.RequestException as req_err:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] Request exception occurred: {req_err}, Time: {elapsed_time:.2f}s")
            return "Gemini Insight: Network error while trying to reach context API."
        except Exception as e:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] Unexpected error occurred: {e}, Time: {elapsed_time:.2f}s")
            return "Gemini Insight: An unexpected error occurred while fetching context."

gemini_api_client = GeminiAPI()
logger.info("[GEMINI] Gemini API client initialized successfully")

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
    
    logger.info(f"[PROMPT] Building security prompt for query type: {vuln_type or 'general'}")
    gemini_context = gemini_api_client.get_context(query) # Retrieve context from Gemini API
    logger.debug(f"[PROMPT] Gemini context integrated: {len(gemini_context)} characters")

    base_context = f"""You are CyberGuard AI, an expert cybersecurity assistant. You help developers understand security vulnerabilities, provide secure coding practices, and analyze potential threats.\n\n---\n\n**Use the following Gemini Insight as your primary context for answering the user's question:**\n{gemini_context}\n\n---\n\nAlways provide:\n- Clear explanations of security concepts\n- Practical code examples (vulnerable and secure versions)\n- Specific mitigation strategies\n- Best practices and recommendations\n\nKeep responses professional, informative, and actionable."""
    
    if vuln_type and vuln_type in VULNERABILITIES:
        vuln = VULNERABILITIES[vuln_type]
        context = f"""\n{base_context}\n\nRelevant vulnerability context:\n- Vulnerability: {vuln['name']}\n- Severity: {vuln['severity']} (CVSS: {vuln['cvss_score']})\n- CVSS Vector: {vuln['cvss_vector']}\n- Description: {vuln['description']}\n- Example vulnerable code: {vuln['example']}\n- Secure implementation: {vuln['secure_example']}\n- Mitigation: {vuln['mitigation']}\n"""
    else:
        context = base_context
    
    prompt = f"""{context}\n\nUser Query: {query}\n\nResponse:"""
    logger.info(f"[PROMPT] Full prompt sent to GPT4All:\n{prompt[:1000]}{'...truncated...' if len(prompt) > 1000 else ''}")
    return prompt

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
    
    # Parse CVSS vector components for explanation
    cvss_vector = vuln.get('cvss_vector', '')
    cvss_explanation = ""
    if cvss_vector:
        # Extract components from CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:N format
        components = cvss_vector.split('/')
        cvss_explanation = f"\n\n**CVSS Vector Components:**\n"
        
        for comp in components[1:]:  # Skip the CVSS:3.1 part
            if ':' in comp:
                key, value = comp.split(':')
                if key == 'AV':
                    cvss_explanation += f"• **Attack Vector ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "Network (remotely exploitable)\n"
                    elif value == 'A': cvss_explanation += "Adjacent (adjacent network required)\n"
                    elif value == 'L': cvss_explanation += "Local (local access required)\n"
                    elif value == 'P': cvss_explanation += "Physical (physical access required)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'AC':
                    cvss_explanation += f"• **Attack Complexity ({key})**: {value} - "
                    if value == 'L': cvss_explanation += "Low (specialized access not required)\n"
                    elif value == 'H': cvss_explanation += "High (significant specialization required)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'PR':
                    cvss_explanation += f"• **Privileges Required ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "None (no privileges required)\n"
                    elif value == 'L': cvss_explanation += "Low (basic privileges required)\n"
                    elif value == 'H': cvss_explanation += "High (significant privileges required)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'UI':
                    cvss_explanation += f"• **User Interaction ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "None (no user interaction required)\n"
                    elif value == 'R': cvss_explanation += "Required (user interaction required)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'S':
                    cvss_explanation += f"• **Scope ({key})**: {value} - "
                    if value == 'U': cvss_explanation += "Unchanged (impact limited to same scope)\n"
                    elif value == 'C': cvss_explanation += "Changed (impact extends beyond scope)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'C':
                    cvss_explanation += f"• **Confidentiality ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "None (no confidentiality impact)\n"
                    elif value == 'L': cvss_explanation += "Low (limited disclosure of data)\n"
                    elif value == 'H': cvss_explanation += "High (significant disclosure of data)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'I':
                    cvss_explanation += f"• **Integrity ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "None (no integrity impact)\n"
                    elif value == 'L': cvss_explanation += "Low (limited modification possible)\n"
                    elif value == 'H': cvss_explanation += "High (significant data modification)\n"
                    else: cvss_explanation += "Unknown\n"
                elif key == 'A':
                    cvss_explanation += f"• **Availability ({key})**: {value} - "
                    if value == 'N': cvss_explanation += "None (no availability impact)\n"
                    elif value == 'L': cvss_explanation += "Low (reduced performance or interruptions)\n"
                    elif value == 'H': cvss_explanation += "High (resource unavailability)\n"
                    else: cvss_explanation += "Unknown\n"
                else:
                    cvss_explanation += f"• **{key}**: {value}\n"
      # Determine language syntax highlighting based on vulnerability type
    lang_syntax = "sql" if vuln_type == "sql_injection" else "javascript" if vuln_type == "xss" or vuln_type == "csrf" else "c" if vuln_type == "buffer_overflow" else "code"
    
    # Create a custom emoji and color per vulnerability type
    vuln_emoji = {
        "sql_injection": "💉",
        "xss": "📜",
        "csrf": "🔄",
        "buffer_overflow": "💾"
    }.get(vuln_type, "🛡️")
    
    # Add relevant OWASP references based on vulnerability type
    owasp_ref = ""
    cwe_ref = ""
    if vuln_type == "sql_injection":
        owasp_ref = "• OWASP A03:2021 Injection\n"
        cwe_ref = "• CWE-89: Improper Neutralization of Special Elements in SQL Commands\n"
    elif vuln_type == "xss":
        owasp_ref = "• OWASP A07:2021 Cross-Site Scripting\n"
        cwe_ref = "• CWE-79: Improper Neutralization of Input During Web Page Generation\n"
    elif vuln_type == "csrf":
        owasp_ref = "• OWASP A05:2021 Security Misconfiguration\n"
        cwe_ref = "• CWE-352: Cross-Site Request Forgery\n"
    elif vuln_type == "buffer_overflow":
        owasp_ref = "• OWASP A08:2021 Software and Data Integrity Failures\n"
        cwe_ref = "• CWE-120: Buffer Copy without Checking Size of Input\n"
    
    if ai_response:
        response = f"""{vuln_emoji} **{vuln['name']} Security Analysis**

{ai_response}

---

**📊 Technical Details:**
- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)
- **Risk Category:** {get_risk_category(vuln['cvss_score'])}
- **CVSS Vector:** {cvss_vector}{cvss_explanation}

**🚨 Vulnerable Code Example:**
```{lang_syntax}
{vuln['example']}
```

**✅ Secure Implementation:**
```{lang_syntax}
{vuln['secure_example']}
```

**🔧 Primary Mitigation:**
{vuln['mitigation']}

**📚 References:**
{owasp_ref}• OWASP Top 10 Security Risks
{cwe_ref}• CWE Database - Common Weakness Enumeration
• NIST Cybersecurity Framework
• MITRE ATT&CK Framework"""
    else:
        # Fallback response if model fails
        response = f"""{vuln_emoji} **{vuln['name']} Security Analysis**

**📝 Description:**
{vuln['description']}

**📊 Risk Assessment:**
- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)
- **Risk Category:** {get_risk_category(vuln['cvss_score'])}
- **CVSS Vector:** {cvss_vector}{cvss_explanation}

**🚨 Vulnerable Code Example:**
```{lang_syntax}
{vuln['example']}
```

**✅ Secure Implementation:**
```{lang_syntax}
{vuln['secure_example']}
```

**🔧 Mitigation Strategies:**
{vuln['mitigation']}

**📚 Security Resources:**
{owasp_ref}• OWASP Top 10 Security Risks
{cwe_ref}• SANS Top 25 Most Dangerous Software Errors
• NIST Cybersecurity Framework Guidelines
• CVE Database and NVD"""
    
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
            "vulnerabilities": "/api/vulnerabilities",
            "stats": "/api/stats",
            "gemini_stats": "/api/gemini-stats"
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
        
        # Manage conversation history with stricter limits to prevent cookie overflow
        if 'conversation' not in session:
            session['conversation'] = []
        
        # Truncate user message and response for session storage to prevent cookie overflow
        user_message_truncated = user_message[:200] if len(user_message) > 200 else user_message
        response_truncated = response[:500] if len(response) > 500 else response
        
        # Add to conversation history with truncated content
        session['conversation'].append({"role": "user", "content": user_message_truncated})
        session['conversation'].append({"role": "assistant", "content": response_truncated})
        
        # Keep only last 2 exchanges to prevent session overflow (4 messages total)
        session['conversation'] = session['conversation'][-4:]  # 4 = 2 exchanges * 2 messages each
        
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

@app.route('/api/gemini-stats', methods=['GET'])
def get_gemini_stats():
    """Get Gemini API usage statistics"""
    return jsonify({
        "gemini_api": {
            "total_calls": gemini_api_client.api_call_count,
            "successful_calls": gemini_api_client.successful_calls,
            "failed_calls": gemini_api_client.failed_calls,
            "success_rate": f"{(gemini_api_client.successful_calls / max(gemini_api_client.api_call_count, 1) * 100):.1f}%",
            "api_key_configured": bool(os.environ.get("GEMINI_API_KEY"))
        }
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
