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

# Use the requested Gemma model
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
        try:
            url = API_URL + self._api_key
            prompt = f"BSL signs to translate: {words}"
            
            data = {
                "system_instruction": {
                    "parts": [{"text": INSTRUCTION}]
                },
                "contents": [
                    {
                        "parts": [{"text": prompt}]
                    }
                ]
            }
            
            req = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"))
            req.add_header("Content-Type", "application/json")
            
            with urllib.request.urlopen(req) as response:
                result_json = json.loads(response.read().decode("utf-8"))
                
                # Parse response
                result = result_json["candidates"][0]["content"]["parts"][0]["text"].strip()
                
                # Remove any wrapping quotes that Gemini might add
                if result.startswith('"') and result.endswith('"'):
                    result = result[1:-1]
                    
                print(f"  ✓ Gemini → \"{result}\"")
                return result
                
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            print(f"  ⚠  Gemini API error ({e.code}): {err_msg}")
            return ' '.join(words)
        except Exception as e:
            print(f"  ⚠  Gemini API error: {e}")
            return ' '.join(words)

