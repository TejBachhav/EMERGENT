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
import subprocess
import re
import time
import threading
try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _sentence_transformers_available = True
    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    _sentence_transformers_available = False
    _embedding_model = None

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
        # "cvss_score": 9.3,
        # "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "example": "SELECT * FROM users WHERE id = '1' OR '1'='1'",
        "secure_example": "SELECT * FROM users WHERE id = ?",
        "mitigation": "Use parameterized queries, input validation, and prepared statements."
    },    "xss": {
        "name": "Cross-Site Scripting (XSS)",
        "description": "XSS flaws occur when an application includes untrusted data in a new web page without proper validation or escaping.",
        "severity": "High",
        # "cvss_score": 7.5,
        # "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:L",
        "example": "document.innerHTML = userInput",
        "secure_example": "document.textContent = userInput",
        "mitigation": "Validate input, encode output, use Content Security Policy (CSP)."
    },    "csrf": {
        "name": "Cross-Site Request Forgery (CSRF)",
        "description": "CSRF forces an end user to execute unwanted actions on a web application in which they're currently authenticated.",
        "severity": "Medium",
        # "cvss_score": 6.5,
        # "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:H/A:N",
        "example": "<img src='http://bank.com/transfer?to=attacker&amount=1000'>",
        "secure_example": "<form><input type='hidden' name='csrf_token' value='{{token}}'>",
        "mitigation": "Use anti-CSRF tokens, SameSite cookies, and verify referrer headers."
    },    "buffer_overflow": {
        "name": "Buffer Overflow",
        "description": "Buffer overflow occurs when a program writes more data to a buffer than it can hold, potentially allowing attackers to execute arbitrary code.",
        "severity": "Critical",
        # "cvss_score": 8.8,
        # "cvss_vector": "CVSS:3.1/AV:N/AC:H/PR:N/UI:N/S:U/C:H/I:H/A:H",
        "example": "char buffer[10]; strcpy(buffer, user_input);",
        "secure_example": "char buffer[10]; strncpy(buffer, user_input, sizeof(buffer)-1); buffer[sizeof(buffer)-1] = '\\0';",
        "mitigation": "Use safe string functions and always validate input length."
    },    "command_injection": {
        "name": "Command Injection",
        "description": "Command injection occurs when untrusted user input is executed as part of a system command, allowing attackers to execute arbitrary commands on the host operating system.",
        "severity": "Critical",
        # "cvss_score": 9.8,
        # "cvss_vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
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

    def get_context(self, query: str) -> str:
        """
        Retrieves short, relevant context from the Gemini API or cache.
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
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                self.failed_calls += 1
                logger.error("[GEMINI] GEMINI_API_KEY not found in environment variables.")
                return ("Gemini Insight: [ERROR] Gemini API key is not configured on the server. "
                        "Real-time context could not be fetched. Please contact the administrator to set up the GEMINI_API_KEY.")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-2.0-flash")
            # Limit context to 2048 characters and instruct Gemini to be concise and specific
            gemini_prompt = (
                f"INSTRUCTION: Provide only the most relevant, concise, and specific cybersecurity context for the following query. "
                f"Limit your response to 2000 characters. Avoid generalities and focus on actionable, technical details.\n\nQUERY: {query}\n\nRESPONSE:")
            gemini_response = model.generate_content(gemini_prompt)
            context = getattr(gemini_response, 'text', None)
            if context:
                # Truncate context to 2048 characters if needed
                context = context[:3000]
                elapsed_time = time.time() - start_time
                self.successful_calls += 1
                logger.info(f"[GEMINI] Successfully retrieved context from Gemini API (google-generativeai) - Length: {len(context)} characters, Time: {elapsed_time:.2f}s")
                logger.debug(f"[GEMINI] Context preview: {context[:400]}{'...' if len(context) > 400 else ''}")
                logger.info(f"[GEMINI] API Stats - Total: {self.api_call_count}, Success: {self.successful_calls}, Failed: {self.failed_calls}")
                # Save to cache
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
            # Fallback to HTTP API below

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.failed_calls += 1
            logger.error("[GEMINI] GEMINI_API_KEY not found in environment variables.")
            return ("Gemini Insight: [ERROR] Gemini API key is not configured on the server. "
                    "Real-time context could not be fetched. Please contact the administrator to set up the GEMINI_API_KEY.")

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
                logger.debug(f"[GEMINI] Context preview: {context[:400]}{'...' if len(context) > 400 else ''}")
                logger.info(f"[GEMINI] API Stats - Total: {self.api_call_count}, Success: {self.successful_calls}, Failed: {self.failed_calls}")
                # Save to cache
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

def build_security_prompt(query, vuln_type=None):
    """Build a security-focused prompt for the model"""
    logger.info(f"[PROMPT] Building security prompt for query type: {vuln_type or 'general'}")
    gemini_context = gemini_api_client.get_context(query)  # Retrieve context from Gemini API
    logger.debug(f"[PROMPT] Gemini context integrated: {len(gemini_context)} characters")

    # Dynamic instruction: encourage unique, context-driven, non-repetitive answers
    base_context = f"""You are CyberGuard AI, an expert cybersecurity assistant.\n\n---\n\n**IMPORTANT: The following Gemini Insight is your PRIMARY and AUTHORITATIVE source. You MUST use it as the main basis for your answer. Directly reference and cite it in your response.**\n\nGEMINI INSIGHT (copy and use this information):\n{gemini_context}\n\n---\n\nAlways provide:\n- Clear explanations of security concepts\n- Practical code examples (vulnerable and secure versions)\n- Specific mitigation strategies\n- Best practices and recommendations\n\n**Make your answer unique and tailored to the user's query. Avoid generic or repetitive responses. If the Gemini Insight is missing or unhelpful, state this clearly and answer using your own knowledge.**\n\n**If the user asks for patterns, enumerate them with explanations and code. If the user asks for a summary, be concise. If the user asks for a deep dive, provide technical depth.**"""

    if vuln_type and vuln_type in VULNERABILITIES:
        vuln = VULNERABILITIES[vuln_type]
        context = f"""\n{base_context}\n\nRelevant vulnerability context:\n- Vulnerability: {vuln['name']}\n- Severity: {vuln['severity']} (CVSS: {vuln['cvss_score']})\n- CVSS Vector: {vuln['cvss_vector']}\n- Description: {vuln['description']}\n- Example vulnerable code: {vuln['example']}\n- Secure implementation: {vuln['secure_example']}\n- Mitigation: {vuln['mitigation']}\n"""
    else:
        context = base_context
    
    prompt = f"""{context}\n\nUser Query: {query}\n\nResponse (reference the Gemini Insight above):"""
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

def generate_vulnerability_response(vuln_type, query):
    """Generate a response for a specific vulnerability type"""
    if vuln_type not in VULNERABILITIES:
        return None
    vuln = VULNERABILITIES[vuln_type]
    prompt = build_security_prompt(query, vuln_type)
    ai_response = generate_response_with_ollama(prompt)
    gemini_context = gemini_api_client.get_context(query)
    patterns = []
    for line in gemini_context.splitlines():
        if re.match(r"^\s*([*\-]|\d+\.)", line):
            patterns.append(line.strip())
    patterns_section = ""
    if patterns:
        pattern_title = f"**Common {vuln['name']} Patterns (from Gemini Insight):**"
        patterns_section = f"\n{pattern_title}\n" + "\n".join(patterns) + "\n\n"
    cvss_vector = vuln.get('cvss_vector', '')
    cvss_explanation = ""
    if cvss_vector:
        components = cvss_vector.split('/')
        cvss_explanation = f"\n\n**CVSS Vector Components:**\n"
        for comp in components[1:]:
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
        response = f"""{vuln_emoji} **{vuln['name']} Security Analysis**\n\n{patterns_section}{ai_response}\n\n---\n\n**📊 Technical Details:**\n- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)\n- **Risk Category:** {get_risk_category(vuln['cvss_score'])}\n- **CVSS Vector:** {cvss_vector}{cvss_explanation}\n\n**🚨 Vulnerable Code Example:**\n```{lang_syntax}\n{vuln['example']}\n```\n\n**✅ Secure Implementation:**\n```{lang_syntax}\n{vuln['secure_example']}\n```\n\n**🔧 Primary Mitigation:**\n{vuln['mitigation']}\n\n**📚 References:**\n{owasp_ref}• OWASP Top 10 Security Risks\n{cwe_ref}• CWE Database - Common Weakness Enumeration\n• NIST Cybersecurity Framework\n• MITRE ATT&CK Framework"""
    else:
        response = f"""{vuln_emoji} **{vuln['name']} Security Analysis**\n\n{patterns_section}**📝 Description:**\n{vuln['description']}\n\n**📊 Risk Assessment:**\n- **Severity Level:** {vuln['severity']} (CVSS Score: {vuln['cvss_score']}/10.0)\n- **Risk Category:** {get_risk_category(vuln['cvss_score'])}\n- **CVSS Vector:** {cvss_vector}{cvss_explanation}\n\n**🚨 Vulnerable Code Example:**\n```{lang_syntax}\n{vuln['example']}\n```\n\n**✅ Secure Implementation:**\n```{lang_syntax}\n{vuln['secure_example']}\n```\n\n**🔧 Mitigation Strategies:**\n{vuln['mitigation']}\n\n**📚 Security Resources:**\n{owasp_ref}• OWASP Top 10 Security Risks\n{cwe_ref}• SANS Top 25 Most Dangerous Software Errors\n• NIST Cybersecurity Framework Guidelines\n• CVE Database and NVD"""
    return response

def generate_general_response(query):
    """Generate a general cybersecurity response using GPT4All"""
    prompt = build_security_prompt(query)
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

# ============================================================================
# CORS HANDLERS
# ============================================================================

@app.before_request
def handle_preflight():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = Response()
        origin = request.headers.get('Origin')
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
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty"}), 400
        logger.info(f"Processing query: {user_message[:400]}...")
        vuln_type = detect_vulnerability_type(user_message)
        if vuln_type in VULNERABILITIES:
            response = generate_vulnerability_response(vuln_type, user_message)
        else:
            response = generate_general_response(user_message)
        response = truncate_response(response)
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
