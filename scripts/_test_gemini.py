import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))

try:
    r = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="Say hello. Reply with just one word.",
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=50,
        ),
    )
    print("SUCCESS:", r.text)
except Exception as e:
    print(f"ERROR TYPE: {type(e).__name__}")
    print(f"ERROR: {e}")
