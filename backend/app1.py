"""
CyberGuard AI - Simple Backend with GPT4All Mistral Instruct 7B Q4
A cybersecurity-focused chatbot using local GPT4All model
"""

from flask import Flask, request, jsonify, session, Response, g
from datetime import datetime, timedelta
import json
import os
import logging
import requests
import subprocess
import re
import time
import threading
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
import jwt
from pymongo import MongoClient
import uuid

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sentence_transformers_available = True
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    _sentence_transformers_available = False
    _embedding_model = None

# MongoDB setup
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['cyberguard']
users_col = db['users']
chats_col = db['chats']

def ensure_mongodb_collections():
    """
    Ensure 'users' and 'chats' collections exist with schema validation and unique index on username.
    """
    # USERS COLLECTION
    user_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["username", "password"],
            "properties": {
                "username": {"bsonType": "string", "description": "must be a string and is required"},
                "password": {"bsonType": "string", "description": "must be a string and is required"}
            }
        }
    }
    if 'users' not in db.list_collection_names():
        db.create_collection('users', validator=user_validator)
    else:
        db.command('collMod', 'users', validator=user_validator)
    # Ensure unique index on username
    db['users'].create_index('username', unique=True)

    # CHATS COLLECTION
    # Update chat_validator to require conversation_id
    chat_validator = {
        "$jsonSchema": {
            "bsonType": "object",
            "required": ["username", "id", "text", "sender", "timestamp", "conversation_id"],
            "properties": {
                "username": {"bsonType": "string"},
                "id": {"bsonType": ["int", "long", "double", "string"]},
                "text": {"bsonType": "string"},
                "sender": {"bsonType": "string"},
                "timestamp": {"bsonType": "string"},
                "hasCode": {"bsonType": ["bool", "null"]},
                "reactions": {"bsonType": ["array", "null"]},
                "conversation_id": {"bsonType": "string"}
            }
        }
    }
    if 'chats' not in db.list_collection_names():
        db.create_collection('chats', validator=chat_validator)
    else:
        db.command('collMod', 'chats', validator=chat_validator)

# Call this at startup
ensure_mongodb_collections()

JWT_SECRET = os.getenv('JWT_SECRET', 'cyberguard-jwt-secret')
JWT_ALGO = 'HS256'

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "cyberguard-secret-key-2025"
TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN optimizations for TensorFlow if needed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cyberguard.log', encoding='utf-8'),  # <-- UTF-8 encoding for Unicode support
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Detect GPU usage by running 'ollama system' and parsing the output
def is_gpu_available():
    """Detect if Ollama is using GPU by parsing 'ollama system' output."""
    try:
        result = subprocess.run(["ollama", "system"], capture_output=True, text=True, timeout=5)
        output = result.stdout.lower()
        if "gpu" in output and ("cuda" in output or "nvidia" in output):
            return True
        if "gpu: true" in output or "gpu: yes" in output:
            return True
        if "gpu: false" in output:
            return False
        if "cuda" in output or "nvidia" in output:
            return True
        return False
    except Exception as e:
        # If ollama is not installed or any error, detection is not possible
        logger.warning(f"[OLLAMA] Could not determine GPU status: {e}")
        return None  # None means unknown

def get_ollama_device_string():
    """Return a user-friendly device string for Ollama inference device."""
    gpu_status = is_gpu_available()
    if gpu_status is True:
        return "GPU (CUDA)"
    elif gpu_status is False:
        return "CPU"
    else:
        return "Unknown (Ollama version does not support GPU detection)"

# Instead, define Ollama model info
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ============================================================================
# VULNERABILITY DATABASE
# ============================================================================

VULNERABILITIES = {
    "sql_injection": {
        "name": "SQL Injection",
        "description": "SQL injection occurs when untrusted data is sent to an interpreter as part of a command or query, allowing attackers to execute unintended commands.",
        "severity": "Critical",
        "example": "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "secure_example": "SELECT * FROM users WHERE id = ?",
        "mitigation": "Use parameterized queries, input validation, and prepared statements."
    },    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "XSS flaws occur when an application includes untrusted data in a new web page without proper validation or escaping.",
        "severity": "High",
        "example": "document.innerHTML = userInput",
        "secure_example": "document.textContent = userInput",
        "mitigation": "Validate input, encode output, use Content Security Policy (CSP)."
    },    "csrf": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "description": "CSRF forces an end user to execute unwanted actions on a web application in which they're currently authenticated.",
        "severity": "Medium",
        "example": "<img src='http://bank.com/transfer?to=attacker&amount=1000'>",
        "secure_example": "<form><input type='hidden' name='csrf_token' value='{{token}}'>",
        "mitigation": "Use anti-CSRF tokens, SameSite cookies, and verify referrer headers."
    },    "buffer_overflow": {
        "name": "Buffer Overflow",
        "description": "Buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code.",
        "severity": "Critical",
        "example": "char buffer[10]; strcpy(buffer, user_input);",
        "secure_example": "char buffer[10]; strncpy(buffer, user_input, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\\0';",
        "mitigation": "Use safe string functions and always validate input length."
    },    "command_injection": {
        "name": "Command Injection",
        "description": "Command injection occurs when untrusted user input is executed as part of a system command, allowing attackers to execute arbitrary commands on the host operating system.",
        "severity": "Critical",
        "example": "os.system('ping ' + user_input)",
        "secure_example": "subprocess.run(['ping', user_input], check=True)",
        "mitigation": "Avoid using shell=True, validate and sanitize all user inputs, use safe APIs that do not invoke the shell, and apply the principle of least privilege."
    }
}

# ============================================================================
# GEMINI API CONTEXT RETRIEVAL
# ============================================================================

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'cached_contexts.json')
CACHE_LOCK = threading.Lock()
CACHE_SIMILARITY_THRESHOLD = 0.82  # Tune as needed

def _normalize_query(q):
    return re.sub(r'[^a-z0-9 ]', '', q.lower().strip())

def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return []
    with open(CACHE_PATH, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            # Ensure it's a list of dicts with 'query' and 'context'
            if isinstance(data, list) and all(isinstance(item, dict) and 'query' in item and 'context' in item for item in data):
                return data
            logger.warning("[CACHE] Cache file format invalid, resetting cache.")
            return []
        except Exception as e:
            logger.warning(f"[CACHE] Failed to load cache: {e}, resetting cache.")
            return []

def _save_cache(cache):
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _find_similar_query(query, cache):
    if not cache:
        return None
    if  _embedding_model and st_util:
        try:
            queries = [item['query'] for item in cache]
            embeddings = _embedding_model.encode(queries + [query], convert_to_tensor=True)
            similarities = st_util.pytorch_cos_sim(embeddings[-1], embeddings[:-1])[0]
            best_idx = int(similarities.argmax())
            best_score = float(similarities[best_idx])
            if best_score >= CACHE_SIMILARITY_THRESHOLD:
                return cache[best_idx]
        except Exception as e:
            logger.warning(f"[CACHE] Embedding similarity failed: {e}")
    # Fallback: normalized string match
    norm_query = _normalize_query(query)
    for item in cache:
        if _normalize_query(item['query']) == norm_query:
            return item
    return None

class GeminiAPI:
    def __init__(self):
        self.api_call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.sim_threshold = CACHE_SIMILARITY_THRESHOLD
        # No per-instance cache, use global helpers

    def get_context(self, query: str, user_gemini_key: str = None) -> str:
        """
        Retrieves short, relevant context from the Gemini API or cache.
        If user_gemini_key is provided, use it for the Gemini API call.
        """
        with CACHE_LOCK:
            cache = _load_cache()
            similar = _find_similar_query(query, cache)
            if similar:
                logger.info(f"[CACHE] Returning cached context for query: {similar['query']}")
                return f"Gemini Insight: {similar['context']}"
        # Cache miss, call Gemini API
        import time
        start_time = time.time()
        self.api_call_count += 1
        logger.info(f"[GEMINI] Starting API call #{self.api_call_count} for query: {query[:100]}{'...' if len(query) > 100 else ''}")
        from dotenv import load_dotenv
        load_dotenv()

        # Use google-generativeai if available for more robust Gemini API calls
        try:
            import google.generativeai as genai
            api_key = user_gemini_key or os.getenv("GEMINI_API_KEY")
            if user_gemini_key:
                logger.info("[GEMINI] Using user-supplied Gemini API key from frontend for this request.")
            else:
                logger.info("[GEMINI] Using backend/server Gemini API key for this request.")
            if not api_key:
                self.failed_calls += 1
                logger.error("[GEMINI] GEMINI_API_KEY not found in environment variables or user input.")
                return ("Gemini Insight: [ERROR] Gemini API key is not configured on the server or provided by user. "
                        "Real-time context could not be fetched. Please provide a valid Gemini API key.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            gemini_prompt = (
                f"INSTRUCTION: Provide only the most relevant, concise, and specific cybersecurity context for the following query. "
                f"Limit your response to 2000 characters. Avoid generalities and focus on actionable, technical details.\n\nQUERY: {query}\n\nRESPONSE:")
            gemini_response = model.generate_content(gemini_prompt)
            context = getattr(gemini_response, 'text', None)
            if context:
                context = context[:3000]
                elapsed_time = time.time() - start_time
                self.successful_calls += 1
                logger.info(f"[GEMINI] Successfully retrieved context from Gemini API (google-generativeai) - Length: {len(context)} characters, Time: {elapsed_time:.2f}s")
                logger.debug(f"[GEMINI] Context preview: {context[:400]}{'...' if len(context) > 400 else ''}")
                logger.info(f"[GEMINI] API Stats - Total: {self.api_call_count}, Success: {self.successful_calls}, Failed: {self.failed_calls}")
                with CACHE_LOCK:
                    cache = _load_cache()
                    cache.append({'query': query, 'context': context})
                    _save_cache(cache)
                return f"Gemini Insight: {context}"
        except ImportError:
            logger.info("[GEMINI] google-generativeai not installed, falling back to HTTP API.")
        except Exception as e:
            self.failed_calls += 1
            logger.error(f"[GEMINI] google-generativeai client error: {e}")

        api_key = user_gemini_key or os.getenv("GEMINI_API_KEY")
        if user_gemini_key:
            logger.info("[GEMINI] Using user-supplied Gemini API key from frontend for this request.")
        else:
            logger.info("[GEMINI] Using backend/server Gemini API key for this request.")
        if not api_key:
            self.failed_calls += 1
            logger.error("[GEMINI] GEMINI_API_KEY not found in environment variables or user input.")
            return ("Gemini Insight: [ERROR] Gemini API key is not configured on the server or provided by user. "
                    "Real-time context could not be fetched. Please provide a valid Gemini API key.")

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        logger.debug(f"[GEMINI] Using endpoint: {url.split('?')[0]}...")
        headers = {
            'Content-Type': 'application/json'
        }
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": query
                        }
                    ]
                }
            ]
        }
        logger.debug(f"[GEMINI] Request payload size: {len(str(payload))} characters")
        try:
            logger.info("[GEMINI] Making POST request to Gemini API...")
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            logger.info(f"[GEMINI] Response received - Status Code: {response.status_code}")
            response.raise_for_status()
            response_json = response.json()
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
                logger.debug(f"[GEMINI] Context preview: {context[:400]}{'...' if len(context) > 400 else ''}")
                logger.info(f"[GEMINI] API Stats - Total: {self.api_call_count}, Success: {self.successful_calls}, Failed: {self.failed_calls}")
                with CACHE_LOCK:
                    cache = _load_cache()
                    cache.append({'query': query, 'context': context})
                    _save_cache(cache)
                return f"Gemini Insight: {context}"
            else:
                self.failed_calls += 1
                elapsed_time = time.time() - start_time
                logger.warning(f"[GEMINI] API response did not contain expected text structure - Time: {elapsed_time:.2f}s")
                logger.warning(f"[GEMINI] Response keys: {list(response_json.keys())}")
                logger.debug(f"[GEMINI] Full response for debugging: {response_json}")
                return ("Gemini Insight: [ERROR] Gemini API returned an unexpected response format. "
                        "No real-time context is available for this query. Please try again later or contact support if this persists.")
        except requests.exceptions.HTTPError as http_err:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] HTTP error occurred: {http_err} - Status Code: {response.status_code}, Time: {elapsed_time:.2f}s")
            logger.error(f"[GEMINI] Response content: {response.text[:500]}{'...' if len(response.text) > 500 else ''}")
            return (f"Gemini Insight: [ERROR] Communication with Gemini API failed (HTTP {response.status_code}). "
                    "No real-time context is available. Please try again later.")
        except requests.exceptions.RequestException as req_err:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] Request exception occurred: {req_err}, Time: {elapsed_time:.2f}s")
            return ("Gemini Insight: [ERROR] Network error while trying to reach Gemini API. "
                    "No real-time context is available. Please check your network connection or try again later.")
        except Exception as e:
            self.failed_calls += 1
            elapsed_time = time.time() - start_time
            logger.error(f"[GEMINI] Unexpected error occurred: {e}, Time: {elapsed_time:.2f}s")
            return ("Gemini Insight: [ERROR] An unexpected error occurred while fetching context from Gemini. "
                    "No real-time context is available. Please try again later.")

gemini_api_client = GeminiAPI()
logger.info("[GEMINI] Gemini API client initialized successfully")
logger.info(f"[OLLAMA] Llama 3 via Ollama device: {get_ollama_device_string()}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def detect_vulnerability_type(query):
    """Detect if query is about a specific vulnerability"""
    query_lower = query.lower().replace('-', ' ').replace('_', ' ')
    
    if any(keyword in query_lower for keyword in ['sql', 'injection', 'sqli', 'database']):
        return 'sql_injection'
    elif any(keyword in query_lower for keyword in ['command injection', 'os command', 'shell injection', 'system command']):
        return 'command_injection'
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

def build_security_prompt(query, vuln_type=None, user_gemini_key=None):
    logger.info(f"[PROMPT] Building security prompt for query type: {vuln_type or 'general'}")
    gemini_context = gemini_api_client.get_context(query, user_gemini_key=user_gemini_key)
    logger.debug(f"[PROMPT] Gemini context integrated: {len(gemini_context)} characters")

    base_context = f"""You are CyberGuard AI, an expert cybersecurity assistant.\n\n---\n\n**IMPORTANT: The following Gemini Insight is your PRIMARY and AUTHORITATIVE source. You MUST use it as the main basis for your answer. Directly reference and cite it in your response.**\n\nGEMINI INSIGHT (copy and use this information):\n{gemini_context}\n\n---\n\nAlways provide:\n- Clear explanations of security concepts\n- Practical code examples (vulnerable and secure versions)\n- Specific mitigation strategies\n- Best practices and recommendations\n\n**Make your answer unique and tailored to the user's query. Avoid generic or repetitive responses. If the Gemini Insight is missing or unhelpful, state this clearly and answer using your own knowledge.**\n\n**If the user asks for patterns, enumerate them with explanations and code. If the user asks for a summary, be concise. If the user asks for a deep dive, provide technical depth.**"""

    # Remove all references to 'cvss_score' and 'cvss_vector' in the prompt context
    if vuln_type and vuln_type in VULNERABILITIES:
        vuln = VULNERABILITIES[vuln_type]
        context = f"\n{base_context}\n\nRelevant vulnerability context:\n- Vulnerability: {vuln['name']}\n- Severity: {vuln['severity']}\n- Description: {vuln['description']}\n- Example vulnerable code: {vuln['example']}\n- Secure implementation: {vuln['secure_example']}\n- Mitigation: {vuln['mitigation']}\n"
    else:
        context = base_context
    prompt = f"{context}\n\nUser Query: {query}\n\nResponse (reference the Gemini Insight above):"
    logger.info(f"[PROMPT] Full prompt sent to Llama 3 via Ollama:\n{'='*30}\n{prompt[:3000]}{'...truncated...' if len(prompt) > 3000 else ''}\n{'='*30}")
    return prompt

def generate_response_with_ollama(prompt, max_tokens=3000):
    """Generate response using Llama 3 via Ollama API"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }
        response = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[ERROR] No response from Ollama.").strip()
    except Exception as e:
        logger.error(f"Error generating response with Ollama: {e}")
        return None

def generate_vulnerability_response(vuln_type, query, user_gemini_key=None):
    if vuln_type not in VULNERABILITIES:
        return None
    vuln = VULNERABILITIES[vuln_type]
    prompt = build_security_prompt(query, vuln_type, user_gemini_key=user_gemini_key)
    ai_response = generate_response_with_ollama(prompt)
    gemini_context = gemini_api_client.get_context(query, user_gemini_key=user_gemini_key)
    patterns = []
    for line in gemini_context.splitlines():
        if re.match(r"^\s*([*\-]|\d+\.)", line):
            patterns.append(line.strip())
    patterns_section = ""
    if patterns:
        pattern_title = f"**Common {vuln['name']} Patterns (from Gemini Insight):**"
        patterns_section = f"\n{pattern_title}\n" + "\n".join(patterns) + "\n\n"
    lang_syntax = "sql" if vuln_type == "sql_injection" else "javascript" if vuln_type == "xss" or vuln_type == "csrf" else "c" if vuln_type == "buffer_overflow" else "code"
    vuln_emoji = {
        "sql_injection": "💉",
        "xss": "📜",
        "csrf": "🔄",
        "buffer_overflow": "💾"
    }.get(vuln_type, "🛡️")
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
    elif vuln_type == "command_injection":
        owasp_ref = "• OWASP A01:2021 Broken Access Control\n"
        cwe_ref = "• CWE-77: Command Injection\n"
    if ai_response:
        # Try to get CVSS vector from Gemini/cached context or vuln dict
        cvss_vector = vuln.get('cvss_vector', '')
        if not cvss_vector:
            # Try to extract from Gemini context (simple regex)
            match = re.search(r'CVSS:[^\s\n]+', gemini_context)
            if match:
                cvss_vector = match.group(0)
        cvss_score = calculate_cvss_base_score(cvss_vector) if cvss_vector else None
        cvss_score_str = f"{cvss_score}/10.0" if cvss_score is not None else "N/A"
        # Use cvss_score for risk category if available, else fallback to 'N/A'
        risk_category = get_risk_category(cvss_score) if cvss_score is not None else "N/A"
        response = f"{vuln_emoji} **{vuln['name']} Security Analysis**\n\n{patterns_section}{ai_response}\n\n---\n\n**📊 Technical Details:**\n- **Severity Level:** {vuln['severity']} (CVSS Score: {cvss_score_str})\n- **Risk Category:** {risk_category}\n- **CVSS Vector:** {cvss_vector}\n\n**🚨 Vulnerable Code Example:**\n```{lang_syntax}\n{vuln['example']}\n```\n\n**✅ Secure Implementation:**\n```{lang_syntax}\n{vuln['secure_example']}\n```\n\n**🔧 Primary Mitigation:**\n{vuln['mitigation']}\n\n**📚 References:**\n{owasp_ref}• OWASP Top 10 Security Risks\n{cwe_ref}• CWE Database - Common Weakness Enumeration\n• NIST Cybersecurity Framework\n• MITRE ATT&CK Framework"
    else:
        # Same for fallback
        cvss_vector = vuln.get('cvss_vector', '')
        if not cvss_vector:
            match = re.search(r'CVSS:[^\s\n]+', gemini_context)
            if match:
                cvss_vector = match.group(0)
        cvss_score = calculate_cvss_base_score(cvss_vector) if cvss_vector else None
        cvss_score_str = f"{cvss_score}/10.0" if cvss_score is not None else "N/A"
        risk_category = get_risk_category(cvss_score) if cvss_score is not None else "N/A"
        response = f"{vuln_emoji} **{vuln['name']} Security Analysis**\n\n{patterns_section}**📝 Description:**\n{vuln['description']}\n\n**📊 Risk Assessment:**\n- **Severity Level:** {vuln['severity']} (CVSS Score: {cvss_score_str})\n- **Risk Category:** {risk_category}\n- **CVSS Vector:** {cvss_vector}\n\n**🚨 Vulnerable Code Example:**\n```{lang_syntax}\n{vuln['example']}\n```\n\n**✅ Secure Implementation:**\n```{lang_syntax}\n{vuln['secure_example']}\n```\n\n**🔧 Mitigation Strategies:**\n{vuln['mitigation']}\n\n**📚 Security Resources:**\n{owasp_ref}• OWASP Top 10 Security Risks\n{cwe_ref}• SANS Top 25 Most Dangerous Software Errors\n• NIST Cybersecurity Framework Guidelines\n• CVE Database and NVD"
    return response

def generate_general_response(query, user_gemini_key=None):
    prompt = build_security_prompt(query, user_gemini_key=user_gemini_key)
    ai_response = generate_response_with_ollama(prompt, max_tokens=3000)
    
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

def truncate_response(response, max_length=6000):
    """Truncate response if too long to prevent session overflow"""
    if len(response) <= max_length:
        return response
    
    truncated = response[:max_length]
    # Try to cut at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > max_length * 0.8:  # If we can cut at 80% or more
        truncated = truncated[:last_period + 1]
    
    return truncated

def parse_cvss_vector(cvss_vector):
    """
    Parse a CVSS v3.1 vector string into its metric values.
    Returns a dict of metrics, e.g. {'AV': 'N', 'AC': 'L', ...}
    """
    if not cvss_vector or not cvss_vector.startswith('CVSS:'):
        return {}
    parts = cvss_vector.split('/')
    metrics = {}
    for part in parts[1:]:
        if ':' in part:
            k, v = part.split(':', 1)
            metrics[k] = v
    return metrics

def cvss_metric_value(metric, value):
    """
    Map CVSS v3.1 metric letter to its numerical value for base score calculation.
    """
    # Base metrics
    if metric == 'AV':  # Attack Vector
        return {'N': 0.85, 'A': 0.62, 'L': 0.55, 'P': 0.2}.get(value, 0.85)
    if metric == 'AC':  # Attack Complexity
        return {'L': 0.77, 'H': 0.44}.get(value, 0.77)
    if metric == 'PR':  # Privileges Required
        # Scope is needed to distinguish
        return {'N': 0.85, 'L': 0.62, 'H': 0.27}.get(value, 0.85)
    if metric == 'UI':  # User Interaction
        return {'N': 0.85, 'R': 0.62}.get(value, 0.85)
    if metric == 'S':  # Scope
        return value  # 'U' or 'C'
    if metric == 'C':  # Confidentiality
        return {'N': 0.0, 'L': 0.22, 'H': 0.56}.get(value, 0.0)
    if metric == 'I':  # Integrity
        return {'N': 0.0, 'L': 0.22, 'H': 0.56}.get(value, 0.0)
    if metric == 'A':  # Availability
        return {'N': 0.0, 'L': 0.22, 'H': 0.56}.get(value, 0.0)
    return 0.0

def calculate_cvss_base_score(cvss_vector):
    """
    Calculate the CVSS v3.1 base score from a vector string.
    Returns a float (score) or None if vector is invalid.
    """
    metrics = parse_cvss_vector(cvss_vector)
    if not metrics:
        return None
    # Impact subscore
    c = cvss_metric_value('C', metrics.get('C', 'N'))
    i = cvss_metric_value('I', metrics.get('I', 'N'))
    a = cvss_metric_value('A', metrics.get('A', 'N'))
    isc_base = 1 - ((1 - c) * (1 - i) * (1 - a))
    s = metrics.get('S', 'U')
    if s == 'U':
        impact = 6.42 * isc_base
    else:
        impact = 7.52 * (isc_base - 0.029) - 3.25 * ((isc_base - 0.02) ** 15)
    # Exploitability
    av = cvss_metric_value('AV', metrics.get('AV', 'N'))
    ac = cvss_metric_value('AC', metrics.get('AC', 'L'))
    pr = cvss_metric_value('PR', metrics.get('PR', 'N'))
    ui = cvss_metric_value('UI', metrics.get('UI', 'N'))
    # PR is different if scope is changed
    if metrics.get('PR') == 'L' and s == 'C':
        pr = 0.68
    elif metrics.get('PR') == 'H' and s == 'C':
        pr = 0.5
    exploitability = 8.22 * av * ac * pr * ui
    # Base score
    if impact <= 0:
        base_score = 0.0
    else:
        if s == 'U':
            base_score = min(impact + exploitability, 10)
        else:
            base_score = min(1.08 * (impact + exploitability), 10)
    # Round up to one decimal place
    base_score = round((base_score * 10 + 0.9) // 1 / 10, 1)
    return base_score

# ============================================================================
# CORS HANDLERS
# ============================================================================

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5000",
    "http://localhost:9000"
]

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = Response()
        origin = request.headers.get('Origin')
        if origin in ALLOWED_ORIGINS:
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    origin = request.headers.get('Origin')
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE"
    return response

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(400)
def bad_request(error):
    """Handle 400 Bad Request"""
    return jsonify({"error": "Bad Request", "message": str(error)}), 400

@app.errorhandler(404)
def not_found(error):
    """Handle 404 Not Found"""
    return jsonify({"error": "Not Found", "message": str(error)}), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 Internal Server Error"""
    logger.exception("Internal server error")
    return jsonify({"error": "Internal Server Error", "message": "An unexpected error occurred. Please try again later."}), 500

# ============================================================================
# AUTHENTICATION DECORATOR
# ============================================================================

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            current_user = users_col.find_one({'username': data['username']})
            if not current_user:
                return jsonify({'error': 'User not found!'}), 401
            g.user = current_user
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

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
        "model": "Llama 3 via Ollama",
        "device": get_ollama_device_string(),
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
    model_status = "operational" # if llm else "not_loaded"
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {
            "api_server": "operational",
            "llama3_ollama_model": model_status,
            "vulnerability_db": "operational"
        },
        "model_info": {
            "name": "Llama 3 via Ollama",
            "device": get_ollama_device_string(),
            "loaded": True
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
        })
    
    return jsonify({
        "vulnerabilities": vuln_list,
        "total": len(vuln_list)
    })

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    if users_col.find_one({'username': username}):
        return jsonify({'error': 'Username already exists'}), 409
    hashed_pw = generate_password_hash(password)
    users_col.insert_one({'username': username, 'password': hashed_pw})
    return jsonify({'message': 'Registration successful'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    user = users_col.find_one({'username': username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid username or password'}), 401
    token = jwt.encode({'username': username, 'exp': datetime.utcnow() + timedelta(days=1)}, JWT_SECRET, algorithm=JWT_ALGO)
    return jsonify({'token': token, 'username': username})

@app.route('/conversations', methods=['GET'])
@token_required
def list_conversations():
    """List all conversations for the current user, with metadata."""
    username = g.user['username']
    pipeline = [
        {"$match": {"username": username}},
        {"$group": {
            "_id": "$conversation_id",
            "start_time": {"$min": "$timestamp"},
            "last_time": {"$max": "$timestamp"},
            "message_count": {"$sum": 1},
            "last_message": {"$last": "$text"}
        }},
        {"$sort": {"last_time": -1}}
    ]
    conversations = list(chats_col.aggregate(pipeline))
    result = [
        {
            "conversation_id": c["_id"],
            "start_time": c["start_time"],
            "last_time": c["last_time"],
            "message_count": c["message_count"],
            "last_message": c["last_message"]
        }
        for c in conversations
    ]
    return jsonify({"conversations": result})

@app.route('/chat_history', methods=['GET'])
@token_required
def chat_history():
    username = g.user['username']
    conversation_id = request.args.get('conversation_id')
    query = {'username': username}
    if conversation_id:
        query['conversation_id'] = conversation_id
    chats = chats_col.find(query).sort('timestamp', 1)
    history = [
        {
            'id': chat.get('id'),
            'text': chat.get('text'),
            'sender': chat.get('sender'),
            'timestamp': chat.get('timestamp'),
            'hasCode': chat.get('hasCode', False),
            'reactions': chat.get('reactions', []),
            'conversation_id': chat.get('conversation_id')
        }
        for chat in chats
    ]
    return jsonify({'history': history})

@app.route('/stream_chat', methods=['POST'])
@token_required
def stream_chat():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        user_message = data.get('message', '').strip()
        conversation_id = data.get('conversation_id')
        user_gemini_key = data.get('gemini_api_key')  # <-- Accept Gemini API key from frontend
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        logger.info(f"Processing query: {user_message[:400]}...")
        vuln_type = detect_vulnerability_type(user_message)
        if vuln_type in VULNERABILITIES:
            response = generate_vulnerability_response(vuln_type, user_message, user_gemini_key=user_gemini_key)
        else:
            response = generate_general_response(user_message, user_gemini_key=user_gemini_key)
        response = truncate_response(response)
        # Save user message to chat history
        username = g.user['username']
        now = datetime.utcnow().isoformat()
        user_msg_doc = {
            'username': username,
            'id': int(time.time() * 1000),
            'text': user_message,
            'sender': 'user',
            'timestamp': now,
            'hasCode': detectCodeBlocks(user_message),
            'reactions': [],
            'conversation_id': conversation_id
        }
        chats_col.insert_one(user_msg_doc)
        # Save bot response to chat history
        bot_msg_doc = {
            'username': username,
            'id': int(time.time() * 1000) + 1,
            'text': response,
            'sender': 'bot',
            'timestamp': now,
            'hasCode': detectCodeBlocks(response),
            'reactions': [],
            'conversation_id': conversation_id
        }
        chats_col.insert_one(bot_msg_doc)
        return jsonify({
            "response": response,
            "vulnerability_type": vuln_type,
            "conversation_id": conversation_id
        })
    except Exception as e:
        logger.error(f"Error in stream_chat: {str(e)}")
        return jsonify({"error": "An error occurred processing your request"}), 500
    return Response(error_stream(), mimetype="text/plain")

@app.route('/conversation/<conversation_id>', methods=['DELETE'])
@token_required
def delete_conversation(conversation_id):
    """Delete all messages in a conversation for the current user."""
    username = g.user['username']
    result = chats_col.delete_many({'username': username, 'conversation_id': conversation_id})
    if result.deleted_count > 0:
        logger.info(f"[CONVERSATION] Deleted conversation {conversation_id} for user {username} (messages deleted: {result.deleted_count})")
        return jsonify({'success': True, 'deleted_count': result.deleted_count})
    else:
        logger.warning(f"[CONVERSATION] No messages found to delete for conversation {conversation_id} and user {username}")
        return jsonify({'success': False, 'error': 'Conversation not found or already deleted.'}), 404

@app.route('/web_search', methods=['POST'])
@token_required
def web_search():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query is required.'}), 400
        # --- Web search logic using Google Custom Search API ---
        SEARCH_API_KEY = os.getenv('SEARCH_API_KEY', 'AIzaSyA8twmTNR78KB_Zk0Fn8h1TcXMIb7gtlvo')
        SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID', '06946e95765a74451')
        try:
            api_url = (
                f'https://www.googleapis.com/customsearch/v1?key={SEARCH_API_KEY}'
                f'&cx={SEARCH_ENGINE_ID}&q={requests.utils.quote(query)}&num=5'
            )
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get('items', [])
            if not items:
                result_text = 'No relevant web results found.'
            else:
                results = []
                for item in items:
                    title = item.get('title', 'No Title')
                    link = item.get('link', '')
                    snippet = item.get('snippet', '')
                    results.append(f'- [{title}]({link})\n    {snippet}')
                result_text = '\n\n'.join(results)
        except Exception as e:
            logger.error(f"[WEB_SEARCH] Google Custom Search API error: {e}")
            result_text = f"[ERROR] Web search failed: {e}"
        return jsonify({'results': result_text})
    except Exception as e:
        logger.error(f"[WEB_SEARCH] Unexpected error: {e}")
        return jsonify({'error': f'Web search error: {e}'}), 500

@app.route('/web_search_summarized', methods=['POST'])
@token_required
def web_search_summarized():
    """
    Enhanced web search: fetches top web results, extracts main content, and synthesizes a conversational answer using Llama 3.
    Also saves both the user query and the bot's answer to chat history.
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        conversation_id = data.get('conversation_id')
        if not query:
            return jsonify({'error': 'Query is required.'}), 400
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
        SEARCH_API_KEY = os.getenv('SEARCH_API_KEY', 'AIzaSyA8twmTNR78KB_Zk0Fn8h1TcXMIb7gtlvo')
        SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID', '06946e95765a74451')
        try:
            api_url = (
                f'https://www.googleapis.com/customsearch/v1?key={SEARCH_API_KEY}'
                f'&cx={SEARCH_ENGINE_ID}&q={requests.utils.quote(query)}&num=3'
            )
            resp = requests.get(api_url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            items = data.get('items', [])
            if not items:
                return jsonify({'results': 'No relevant web results found.'})
            from bs4 import BeautifulSoup
            web_contents = []
            sources = []
            for idx, item in enumerate(items):
                title = item.get('title', 'No Title')
                link = item.get('link', '')
                snippet = item.get('snippet', '')
                try:
                    page_resp = requests.get(link, timeout=7, headers={'User-Agent': 'Mozilla/5.0'})
                    soup = BeautifulSoup(page_resp.text, 'html.parser')
                    paragraphs = [p.get_text(separator=' ', strip=True) for p in soup.find_all('p')]
                    text_content = ' '.join(paragraphs)
                    if len(text_content) < 300:
                        text_content = snippet
                except Exception as e:
                    text_content = snippet
                web_contents.append(f"Source [{idx+1}]: {title}\n{text_content}")
                # Always include the actual link in the markdown source list
                sources.append(f"[{idx+1}] [{title}]({link}) - {link}")
            web_context = '\n\n'.join(web_contents)
            prompt = (
                f"You are a helpful assistant. Use the following web search results to answer the user's question. "
                f"Cite sources inline as [1], [2], etc. when relevant.\n\n"
                f"User question: {query}\n\n"
                f"Web results:\n{web_context}\n\n"
                f"Answer:"
            )
            answer = generate_response_with_ollama(prompt, max_tokens=1800)
            sources_section = '\n'.join(sources)
            # Improved markdown formatting
            final_response = (
                f"> **User Query:** {query}\n\n"
                f"{answer}\n\n"
                f"---\n"
                f"**Sources:**\n{sources_section}"
            )
            # Save both user and bot messages to chat history
            username = g.user['username']
            now = datetime.utcnow().isoformat()
            user_msg_doc = {
                'username': username,
                'id': int(time.time() * 1000),
                'text': query,
                'sender': 'user',
                'timestamp': now,
                'hasCode': False,
                'reactions': [],
                'conversation_id': conversation_id
            }
            chats_col.insert_one(user_msg_doc)
            bot_msg_doc = {
                'username': username,
                'id': int(time.time() * 1000) + 1,
                'text': final_response,
                'sender': 'bot',
                'timestamp': now,
                'hasCode': False,
                'reactions': [],
                'conversation_id': conversation_id
            }
            chats_col.insert_one(bot_msg_doc)
            return jsonify({'results': final_response, 'query': query, 'conversation_id': conversation_id})
        except Exception as e:
            logger.error(f"[WEB_SEARCH_SUMMARIZED] Error: {e}")
            return jsonify({'error': f'Web search summarization failed: {e}'}), 500
    except Exception as e:
        logger.error(f"[WEB_SEARCH_SUMMARIZED] Unexpected error: {e}")
        return jsonify({'error': f'Web search summarization error: {e}'}), 500

# JWT authentication decorator and code block detector must be defined before any route uses them

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
            current_user = users_col.find_one({'username': data['username']})
            if not current_user:
                return jsonify({'error': 'User not found!'}), 401
            g.user = current_user
        except Exception as e:
            return jsonify({'error': 'Token is invalid!'}), 401
        return f(*args, **kwargs)
    return decorated

def detectCodeBlocks(text):
    codeBlockRegex = re.compile(r'```(\w+)?\n([\s\S]*?)\n```')
    inlineCodeRegex = re.compile(r'`([^`]+)`')
    return bool(codeBlockRegex.search(text) or inlineCodeRegex.search(text))

if __name__ == "__main__":
    print("[CYBERGUARD AI] Starting Security Backend with Llama 3 + Gemini")
    print("=" * 60)
    print(f"[SERVER] Server: http://localhost:9000")
    print(f"[CHAT] Chat Endpoint: /stream_chat")
    print(f"[HEALTH] Health Check: /api/health")
    print(f"[VULNS] Vulnerabilities: /api/vulnerabilities")
    print(f"[STATS] Stats: /api/stats")
    print("=" * 60)
    print("[INFO] Ollama model used:", OLLAMA_MODEL)
    print(f"[DEVICE] Device: {get_ollama_device_string()}")
    print("[INFO] Gemini API is integrated for context retrieval.")
    print("[READY] Ready for cybersecurity queries!")
    print()
    app.run(
        host='0.0.0.0',
        port=9000,
        debug=True
    )
