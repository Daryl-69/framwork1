import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

try:
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello. Reply with just one word."}],
        temperature=0.5,
        max_tokens=50,
    )
    print("SUCCESS:", r.choices[0].message.content)
except Exception as e:
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR: {e}")
