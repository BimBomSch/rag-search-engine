import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key and not base_url:
    raise RuntimeError("OPENAI_API_KEY, BASE_URL environment variables not set")

client = OpenAI(base_url=base_url, api_key=api_key,)
model = "google/gemma-3-27b"

def spell_correct(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
    Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
    Preserve punctuation and capitalization unless a change is required for a typo fix.
    If there are no spelling errors, or if you're unsure, output the original query unchanged.
    Output only the final query text, nothing else.
    User query: "{query}"
    """

    response = client.responses.create(model=model, input=prompt,)
    corrected = (response.output_text or "").strip().strip('"')
    return corrected if corrected else query


def enhance_query(query: str, method: Optional[str] = None) -> str:
    match method:
        case "spell":
            return spell_correct(query)
        case _:
            return query








