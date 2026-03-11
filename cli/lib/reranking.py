import os
from time import sleep

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
base_url = os.environ.get("BASE_URL")
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key and not base_url:
    raise RuntimeError("OPENAI_API_KEY, BASE_URL environment variables not set")

client = OpenAI(base_url=base_url, api_key=api_key,)
model = "google/gemma-3-27b"


def llm_rerank_individual(
    query: str, documents: list[dict], limit: int = 5
) -> list[dict]:
    scored_docs = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        response = client.responses.create(model=model, input=prompt,)
        score_text = (response.output_text or "").strip()
        score = int(score_text)
        scored_docs.append({**doc, "individual_score": score})
        sleep(0.1)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs[:limit]


def rerank(
    query: str, documents: list[dict], method: str = "batch", limit: int = 5
) -> list[dict]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    else:
        return documents[:limit]
