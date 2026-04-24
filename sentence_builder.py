"""
Gemini API-powered BSL sentence builder.

Uses pure Python standard libraries (urllib.request) to avoid 
any pip dependency conflicts with TensorFlow.

API key setup:
  Create a file at `bsl_data/gemini_api_key.txt`
  Paste your key on the first line.
  Get a free key at: https://aistudio.google.com
"""

import os
import json
import urllib.request
import urllib.error

# ── API key discovery ──────────────────────────────────────────────────────────
KEY_FILE     = os.path.join("bsl_data", "gemini_api_key.txt")
KEY_FILE_ALT = os.path.join("bsl_data", "gemini_api_key.txt.txt")  # Windows double-ext

def _load_api_key() -> str | None:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key
    for path in (KEY_FILE, KEY_FILE_ALT):
        if os.path.exists(path):
            with open(path, "r") as f:
                key = f.read().strip()
            if key:
                return key
    return None

# Use the ultra-fast 8B model which has the highest rate limits on the free tier
GEMINI_MODEL = "gemma-4-26b-a4b-it"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="

# ── System Instruction ─────────────────────────────────────────────────────────
INSTRUCTION = """\
You are a British Sign Language (BSL) to English translator.

BSL grammar rules you must apply:
- BSL uses Topic-Comment word order, which differs from English.
- BSL omits function words — YOU must add "is", "are", "a", "the", "do", etc.
- The signed words may arrive in any order; find the most natural English sentence.

Examples (the input order can be shuffled):
  [your, name, what]   → "What is your name?"
  [home, I, go]        → "I am going home."
  [hello, my, name]    → "Hello, my name is..."
  [good, morning]      → "Good morning."
  [how, you]           → "How are you?"


STRICT RULES:
- Print the sentence first...before anything else.
- Output ONLY the final English sentence.
- No explanation, no alternatives, no label.
- Use correct capitalisation and punctuation.
- Keep it concise and natural.\
"""

class SentenceBuilder:
    def __init__(self):
        self._model_name = GEMINI_MODEL
        self._ready      = False
        self._api_key    = _load_api_key()

        if not self._api_key:
            print("  ✗ No Gemini API key found.")
            print(f"    → Create '{KEY_FILE}' and paste your key inside.")
            return

        self._ready = True
        print(f"  ✓ Gemini API connected via REST ({self._model_name}) — sentence builder ready")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def loading(self) -> bool:
        return False

    def build(self, words: list) -> str:
        if not words:
            return ""
        if not self._ready:
            return ' '.join(words)

        print(f"  → Gemini ← {words}")
        url = API_URL + self._api_key
        
        data = {
            "system_instruction": {
                "parts": [{"text": INSTRUCTION}]
            },
            "contents": [
                {
                    "role": "user", 
                    "parts": [{"text": "BSL signs to translate: ['home', 'I', 'go']\n\nReturn EXACTLY and ONLY the final english sentence. No thinking, no markdown, no bullet points, no 'Result:' labels."}]
                },
                {
                    "role": "model", 
                    "parts": [{"text": "I am going home."}]
                },
                {
                    "role": "user",
                    "parts": [{"text": f"BSL signs to translate: {words}\n\nReturn EXACTLY and ONLY the final english sentence. No thinking, no markdown, no bullet points, no 'Result:' labels."}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 64
            }
        }
        
        encoded_data = json.dumps(data).encode("utf-8")
        
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=encoded_data)
                req.add_header("Content-Type", "application/json")
                
                with urllib.request.urlopen(req) as response:
                    result_json = json.loads(response.read().decode("utf-8"))
                    
                    # Parse response
                    result = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                    
                    # --- HOTFIX FOR GEMMA 4 CHAIN-OF-THOUGHT ---
                    # If the model ignores the instruction and prints paragraphs with bullet points,
                    # simply grab the very final phrase which usually contains the sentence we want.
                    if '\n' in result:
                        lines = [line.strip() for line in result.split('\n') if line.strip()]
                        if lines:
                            result = lines[-1]
                            
                    # Remove markdown elements e.g., `* ` or `- `
                    if result.startswith('*   '):
                        result = result[4:]
                    elif result.startswith('* '):
                        result = result[2:]
                        
                    # Remove any wrapping quotes that Gemini might add
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                        
                    print(f"  ✓ Gemini → \"{result}\"")
                    return result
                    
            except urllib.error.HTTPError as e:
                import time
                err_msg = e.read().decode("utf-8")
                
                if e.code == 429:
                    print(f"  ⚠  Rate limited (429). Retrying in 2 seconds (Attempt {attempt+1}/3)...")
                    time.sleep(2)
                    continue
                else:
                    print(f"  ⚠  Gemini API error ({e.code}): {err_msg}")
                    return ' '.join(words)
            except Exception as e:
                print(f"  ⚠  Gemini API error: {e}")
                return ' '.join(words)
                
        return ' '.join(words)
